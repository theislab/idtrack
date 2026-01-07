#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

import math
import time
import typing as _t
from inspect import signature

import pandas as pd
from tqdm import tqdm

from idtrack._external_mappers._constants import _GP_NS, _GP_INPUT_NAMESPACES
from idtrack._external_mappers._utils import (
    _chunker,
    _empty_result,
    _ensure_all_inputs,
    _is_bare_numeric,
    _json,
    _suppress_stdout_stderr,
    _unique_not_null,
    canonical_db,
    canonical_species,
    logger,
    strip_version,
)


def _gp_target_candidates(outp: str) -> list[str]:
    """
    Return an ordered list of g:Profiler target_namespace candidates
    for a given canonical output DB.
    """
    outp = canonical_db(outp)

    if outp == "uniprot":
        candidates = [
            "UNIPROTSPTREMBL_ACC",
            "UNIPROTSWISSPROT_ACC",
            "UNIPROTSPTREMBL",
            "UNIPROTSWISSPROT",
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for ns in candidates:
            if ns and ns not in seen:
                seen.add(ns)
                ordered.append(ns)
        return ordered

    base = _GP_NS.get(outp)
    return [base] if base else []


def _extract_namespace_tokens(raw: _t.Any) -> set[str]:
    """Normalize the g:Profiler `namespaces` field into uppercase tokens."""
    if raw is None:
        return set()

    if isinstance(raw, (list, tuple, set)):
        iterable = raw
    elif isinstance(raw, str):
        iterable = raw.split(",")
    else:
        try:
            if pd.isna(raw):
                return set()
        except Exception:
            pass
        iterable = [raw]

    tokens: set[str] = set()
    for item in iterable:
        if item is None:
            continue
        try:
            if pd.isna(item):
                continue
        except Exception:
            pass
        token = str(item).strip()
        if token:
            tokens.add(token.upper())
    return tokens


def map_with_gprofiler(
    ids: _t.Iterable[str],
    input_db: str,
    output_db: str,
    *,
    species: str = "hsapiens",
    chunk_size: int = 1000,
    pause: float = 0.2,
    max_retries: int = 3,
    strip_versions: bool = True,
    show_progress: bool = True,
    suppress_method_verbosity: bool = True,
    strict_input_db: bool = False,
) -> pd.DataFrame:
    """Map IDs via g:Profiler (gprofiler-official)."""
    try:
        from gprofiler import GProfiler  # type: ignore
    except Exception as e:
        raise RuntimeError("gprofiler-official is not installed. Try: pip install gprofiler-official") from e

    species = canonical_species(species)
    inp = canonical_db(input_db)
    outp = canonical_db(output_db)
    output_input_db_value: str | None = inp if strict_input_db else None

    namespace_filter: _t.Callable[[_t.Any], bool] | None = None
    if strict_input_db:
        allowed = _GP_INPUT_NAMESPACES.get(inp)
        if not allowed:
            raise ValueError(f"strict_input_db is not supported for input database {inp!r}")
        allowed_upper = {a.upper() for a in allowed}

        def namespace_filter(value: _t.Any, allowed_tokens: set[str] = allowed_upper) -> bool:
            return bool(_extract_namespace_tokens(value) & allowed_tokens)

    target_candidates = _gp_target_candidates(outp)
    if not target_candidates:
        raise ValueError(f"g:Profiler: unsupported target namespace for {outp!r}")

    clean_ids = [strip_version(i) if strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)
    if not uniq_ids:
        ensured = _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gprofiler", release_used=None)
        ensured = ensured.copy()
        ensured["input_db"] = output_input_db_value
        return ensured

    gp = GProfiler(return_dataframe=True)

    try:
        sig_params = set(signature(gp.convert).parameters)
    except Exception:
        sig_params = set()

    if "target_namespace" in sig_params:
        target_key = "target_namespace"
    elif "target" in sig_params:
        target_key = "target"
    else:
        target_key = "target_namespace"

    numeric_param: str | None = None
    if "numeric_namespace" in sig_params:
        numeric_param = "numeric_namespace"
    elif "numeric_ns" in sig_params:
        numeric_param = "numeric_ns"

    base_kwargs: dict[str, _t.Any] = {"organism": species}

    if numeric_param is not None and inp == "entrez_gene" and all(_is_bare_numeric(x) for x in uniq_ids):
        base_kwargs[numeric_param] = "ENTREZGENE_ACC"

    last_error: Exception | None = None
    selected_frames: list[pd.DataFrame] | None = None

    for target_ns in target_candidates:
        logger.debug("g:Profiler: trying target namespace %r", target_ns)
        frames: list[pd.DataFrame] = []
        any_non_null = False
        n_chunks = math.ceil(len(uniq_ids) / chunk_size)
        with tqdm(
            total=len(uniq_ids),
            desc=f"gprofiler",
            mininterval=0.25,
            disable=not show_progress,
            ncols=100,
            unit="ids",
        ) as progress:
            for i, chunk in enumerate(_chunker(uniq_ids, chunk_size), start=1):
                logger.debug(
                    "g:Profiler: querying chunk %d/%d (n=%d, target_ns=%s)",
                    i,
                    n_chunks,
                    len(chunk),
                    target_ns,
                )
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        kwargs = dict(base_kwargs)
                        kwargs[target_key] = target_ns
                        kwargs["query"] = chunk
                        with _suppress_stdout_stderr(suppress_method_verbosity):
                            df = gp.convert(**kwargs)

                        if df is not None and not df.empty and namespace_filter is not None:
                            if "namespaces" not in df.columns:
                                raise RuntimeError("strict_input_db requires g:Profiler to return a 'namespaces' column")
                            mask = df["namespaces"].apply(namespace_filter)
                            df = df[mask]

                        if df is None or df.empty:
                            frames.append(pd.DataFrame(columns=["input_id", "output_id", "metadata_json"]))
                        else:
                            keep_cols = ["incoming", "converted"]
                            extra_cols = [c for c in df.columns if c not in keep_cols]
                            keep = df[keep_cols].rename(columns={"incoming": "input_id", "converted": "output_id"})
                            keep = keep.drop_duplicates()

                            if extra_cols:
                                meta_records: list[tuple[_t.Any, dict[str, _t.Any]]] = []
                                for _, row in df[["incoming"] + extra_cols].drop_duplicates().iterrows():
                                    q = row["incoming"]
                                    meta = {c: row[c] for c in extra_cols}
                                    meta_records.append((q, meta))
                                meta_df = pd.DataFrame(meta_records, columns=["input_id", "meta"])
                                keep = keep.merge(meta_df, on="input_id", how="left")
                                keep["metadata_json"] = keep["meta"].apply(
                                    lambda m: _json(m if isinstance(m, dict) else {})
                                )
                                keep = keep.drop(columns=["meta"])
                            else:
                                keep["metadata_json"] = _json({})

                            frames.append(keep)
                            if "output_id" in keep.columns and keep["output_id"].notna().any():
                                any_non_null = True

                        break

                    except Exception as e:
                        last_error = e
                        logger.warning(
                            "g:Profiler batch failed for target %s (attempt %d): %s",
                            target_ns,
                            attempt,
                            e,
                        )
                        if attempt >= max_retries:
                            err_meta = _json({"error": str(e), "target_namespace": target_ns})
                            frames.append(
                                pd.DataFrame(
                                    {
                                        "input_id": chunk,
                                        "output_id": [None] * len(chunk),
                                        "metadata_json": [err_meta] * len(chunk),
                                    }
                                )
                            )
                            break
                        time.sleep(pause * attempt)

                progress.update(len(chunk))
                time.sleep(pause)

        if frames and any_non_null:
            selected_frames = frames
            break

    if selected_frames is None:
        if last_error is not None:
            base = pd.DataFrame(
                {
                    "input_id": [str(x) for x in clean_ids],
                    "input_db": inp,
                    "output_id": [None] * len(clean_ids),
                    "output_db": outp,
                    "method": "gprofiler",
                    "release_used": None,
                    "metadata_json": _json({"error": str(last_error)}),
                }
            )
            from ._utils import _add_mapping_column  # avoid circular import at top

            base = _add_mapping_column(base)
            base = base.copy()
            base["input_db"] = output_input_db_value
            return base[
                [
                    "input_id",
                    "input_db",
                    "output_id",
                    "output_db",
                    "method",
                    "release_used",
                    "mapping",
                    "metadata_json",
                ]
            ]

        ensured = _ensure_all_inputs(
            _empty_result(),
            clean_ids,
            inp,
            outp,
            "gprofiler",
            release_used=None,
        )
        ensured = ensured.copy()
        ensured["input_db"] = output_input_db_value
        return ensured

    out = pd.concat(selected_frames, ignore_index=True)

    if "metadata_json" not in out.columns:
        out["metadata_json"] = _json({})

    out["input_db"] = output_input_db_value
    out["output_db"] = outp
    out["method"] = "gprofiler"
    out["release_used"] = None

    out = _ensure_all_inputs(out, clean_ids, inp, outp, "gprofiler", release_used=None)
    out["input_db"] = output_input_db_value
    out = out.drop_duplicates(["input_id", "output_id", "input_db", "output_db", "method", "release_used"])

    return out[
        [
            "input_id",
            "input_db",
            "mapping",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "metadata_json",
        ]
    ]
