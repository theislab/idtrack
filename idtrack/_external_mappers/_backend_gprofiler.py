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

from idtrack._external_mappers._constants import _GP_INPUT_NAMESPACES, _GP_NS
from idtrack._external_mappers._utils import (
    _add_mapping_column,
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
    raise_missing_dependency,
    strip_version,
)

# UniProt target namespace candidates in priority order
_UNIPROT_TARGET_CANDIDATES = (
    "UNIPROTSPTREMBL_ACC",
    "UNIPROTSWISSPROT_ACC",
    "UNIPROTSPTREMBL",
    "UNIPROTSWISSPROT",
)


def _gp_target_candidates(outp: str) -> list[str]:
    """Return g:Profiler target_namespace candidates for a canonical output database.

    The returned list is ordered by preference (first hit wins).

    Args:
        outp: Canonical output database name.

    Returns:
        list[str]: Target namespace candidates in preference order.
    """
    outp = canonical_db(outp)
    if outp == "uniprot":
        return list(_UNIPROT_TARGET_CANDIDATES)
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
        except (TypeError, ValueError):
            # pd.isna() can fail on certain types; treat as non-NA value
            pass
        iterable = [raw]

    tokens: set[str] = set()
    for item in iterable:
        if item is None:
            continue
        try:
            if pd.isna(item):
                continue
        except (TypeError, ValueError):
            # pd.isna() can fail on certain types; treat as non-NA value
            pass
        token = str(item).strip()
        if token:
            tokens.add(token.upper())
    return tokens


def _build_metadata_column(df: pd.DataFrame, extra_cols: list[str]) -> pd.Series:
    """Build a metadata_json column from extra columns.

    Args:
        df: Input DataFrame containing columns referenced by ``extra_cols``.
        extra_cols: Column names to include in the JSON metadata per row.

    Returns:
        pd.Series: JSON-encoded metadata aligned with ``df.index``.
    """
    if not extra_cols:
        return pd.Series([_json({})] * len(df), index=df.index)

    def row_to_json(row: pd.Series) -> str:
        meta = {col: row[col] for col in extra_cols if col in row.index}
        return _json(meta)

    return df[extra_cols].apply(row_to_json, axis=1)


def _process_gprofiler_response(
    df: pd.DataFrame | None,
    namespace_filter: _t.Callable[[_t.Any], bool] | None,
) -> tuple[pd.DataFrame, bool]:
    """Process a g:Profiler convert() response into standardized format.

    Args:
        df: Raw g:Profiler response DataFrame (or ``None``).
        namespace_filter: Optional predicate to filter the ``namespaces`` column when enforcing
            strict input-db behavior.

    Returns:
        tuple[pd.DataFrame, bool]: ``(processed_df, has_non_null_outputs)``.

    Raises:
        RuntimeError: If ``namespace_filter`` is provided but the response lacks a ``namespaces`` column.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["input_id", "output_id", "metadata_json"]), False

    # Apply namespace filter if strict_input_db is enabled
    if namespace_filter is not None:
        if "namespaces" not in df.columns:
            raise RuntimeError("strict_input_db requires g:Profiler to return a 'namespaces' column")
        mask = df["namespaces"].apply(namespace_filter)
        df = df[mask]
        if df.empty:
            return pd.DataFrame(columns=["input_id", "output_id", "metadata_json"]), False

    # Extract input/output columns
    keep_cols = {"incoming", "converted"}
    extra_cols = [c for c in df.columns if c not in keep_cols]

    result = (
        df[["incoming", "converted"]]
        .rename(columns={"incoming": "input_id", "converted": "output_id"})
        .drop_duplicates()
    )

    # Build metadata from extra columns
    if extra_cols:
        meta_df = df[["incoming"] + extra_cols].drop_duplicates()
        meta_df = meta_df.rename(columns={"incoming": "input_id"})
        meta_df["metadata_json"] = _build_metadata_column(meta_df, extra_cols)
        meta_df = meta_df[["input_id", "metadata_json"]].drop_duplicates(subset=["input_id"])
        result = result.merge(meta_df, on="input_id", how="left")
        result["metadata_json"] = result["metadata_json"].fillna(_json({}))
    else:
        result["metadata_json"] = _json({})

    has_outputs = result["output_id"].notna().any()
    return result, has_outputs


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
    """Map IDs via g:Profiler (gprofiler-official).

    Args:
        ids: Input identifiers to map.
        input_db: Database/namespace of input IDs (e.g. ``"ensembl_gene"``, ``"hgnc_symbol"``).
        output_db: Target database/namespace (e.g. ``"uniprot"``, ``"entrez_gene"``).
        species: Species code (e.g. ``"hsapiens"``, ``"mmusculus"``, ``"sscrofa"``).
        chunk_size: Number of IDs per API request.
        pause: Seconds to pause between requests.
        max_retries: Maximum retry attempts per chunk on failure.
        strip_versions: Strip version suffixes from Ensembl/RefSeq IDs.
        show_progress: Display progress bar.
        suppress_method_verbosity: Suppress stdout/stderr from the gprofiler library.
        strict_input_db: If True, filter results to only include mappings where the input namespace matches
            the expected ``input_db``.

    Returns:
        pd.DataFrame: Standardized mapping DataFrame.

    Raises:
        ValueError: If ``strict_input_db`` is enabled for an unsupported input database.
    """
    try:
        from gprofiler import GProfiler  # type: ignore
    except ImportError as e:
        raise_missing_dependency("gprofiler-official", feature="g:Profiler ID mapping backend", original_error=e)

    species = canonical_species(species)
    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    # Setup namespace filter for strict_input_db mode
    namespace_filter: _t.Callable[[_t.Any], bool] | None = None
    if strict_input_db:
        allowed = _GP_INPUT_NAMESPACES.get(inp)
        if not allowed:
            raise ValueError(f"strict_input_db is not supported for input database {inp!r}")
        allowed_upper = {a.upper() for a in allowed}

        def _namespace_filter(ns: _t.Any, *, allowed: set[str] = allowed_upper) -> bool:
            return bool(_extract_namespace_tokens(ns) & allowed)

        namespace_filter = _namespace_filter

    # Validate output namespace
    target_candidates = _gp_target_candidates(outp)
    if not target_candidates:
        raise ValueError(f"g:Profiler: unsupported target namespace for {outp!r}")

    # Prepare input IDs
    clean_ids = [strip_version(i) if strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)

    # Handle empty input
    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gprofiler", release_used=None)

    # Initialize g:Profiler client
    gp = GProfiler(return_dataframe=True)

    # Detect API parameter names (handles different gprofiler-official versions)
    try:
        sig_params = set(signature(gp.convert).parameters)
    except (ValueError, TypeError):
        # signature() can fail on built-in functions or unusual callables
        sig_params = set()

    target_key = "target_namespace" if "target_namespace" in sig_params else "target"
    numeric_param = (
        "numeric_namespace"
        if "numeric_namespace" in sig_params
        else "numeric_ns" if "numeric_ns" in sig_params else None
    )

    # Base kwargs for all requests
    base_kwargs: dict[str, _t.Any] = {"organism": species}

    # Handle numeric Entrez IDs
    if numeric_param is not None and inp == "entrez_gene" and all(_is_bare_numeric(x) for x in uniq_ids):
        base_kwargs[numeric_param] = "ENTREZGENE_ACC"

    # Try each target namespace candidate until one returns results
    last_error: Exception | None = None
    selected_frames: list[pd.DataFrame] | None = None

    for target_ns in target_candidates:
        logger.debug("g:Profiler: trying target namespace %r", target_ns)
        frames: list[pd.DataFrame] = []
        any_non_null = False
        n_chunks = math.ceil(len(uniq_ids) / chunk_size)

        with tqdm(
            total=len(uniq_ids),
            desc="gprofiler",
            mininterval=0.25,
            disable=not show_progress,
            ncols=100,
            unit="ids",
        ) as progress:
            for chunk_idx, chunk in enumerate(_chunker(uniq_ids, chunk_size), start=1):
                logger.debug(
                    "g:Profiler: querying chunk %d/%d (n=%d, target_ns=%s)",
                    chunk_idx,
                    n_chunks,
                    len(chunk),
                    target_ns,
                )

                # Retry loop for this chunk
                for attempt in range(1, max_retries + 1):
                    try:
                        kwargs = {**base_kwargs, target_key: target_ns, "query": chunk}
                        with _suppress_stdout_stderr(suppress_method_verbosity):
                            df = gp.convert(**kwargs)

                        result, has_outputs = _process_gprofiler_response(df, namespace_filter)
                        frames.append(result)
                        if has_outputs:
                            any_non_null = True
                        break

                    except Exception as e:
                        last_error = e
                        logger.warning(
                            "g:Profiler batch failed for target %s (attempt %d/%d): %s",
                            target_ns,
                            attempt,
                            max_retries,
                            e,
                        )
                        if attempt >= max_retries:
                            # Record failure for this chunk
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
                        else:
                            time.sleep(pause * attempt)

                progress.update(len(chunk))
                time.sleep(pause)

        # If we got any non-null results, use this target namespace
        if frames and any_non_null:
            selected_frames = frames
            break

    # Build final output
    if selected_frames is None:
        # No successful results from any target namespace
        if last_error is not None:
            # Return error information
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
            base = _add_mapping_column(base)
            return base[
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
        # Empty results
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gprofiler", release_used=None)

    # Combine all frames
    out = pd.concat(selected_frames, ignore_index=True)

    # Ensure metadata column exists
    if "metadata_json" not in out.columns:
        out["metadata_json"] = _json({})

    # Add standard columns
    out["input_db"] = inp
    out["output_db"] = outp
    out["method"] = "gprofiler"
    out["release_used"] = None

    # Ensure all inputs are represented and add mapping column
    out = _ensure_all_inputs(out, clean_ids, inp, outp, "gprofiler", release_used=None)

    # Remove duplicates
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
