#!/usr/bin/env python3
"""gget backend for ID mapping.

This module provides the map_with_gget() function for querying
the gget.info API (Ensembl REST-backed) to convert biological identifiers.
"""

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

import math
import time
import typing as _t

import pandas as pd
from tqdm import tqdm

from idtrack._external_mappers._constants import _ENSEMBL_INPUT_DB
from idtrack._external_mappers._utils import (
    _chunker,
    _empty_result,
    _ensure_all_inputs,
    _json,
    _suppress_stdout_stderr,
    _unique_not_null,
    canonical_db,
    canonical_species,
    logger,
    raise_missing_dependency,
    strip_version,
)


def _gget_extract(df: pd.DataFrame, outp: str) -> pd.DataFrame:
    """Normalize gget.info() output into standardized (input_id, output_id) format.

    Args:
        df: Raw DataFrame from gget.info().
        outp: Target database type.

    Returns:
        DataFrame with ``input_id`` and ``output_id`` columns.
    """
    rename = {
        "id": "gene_id",
        "gene": "gene_id",
        "ensembl_id": "gene_id",
        "name": "gene_name",
        "display_name": "gene_name",
        "symbol": "gene_name",
        "gene_symbol": "gene_name",
        "primary_gene_name": "gene_name",
        "entrez": "entrez_id",
        "entrezgene": "entrez_id",
        "entrez_gene": "entrez_id",
        "ncbi_gene_id": "entrez_id",
        "uniprot": "uniprot_id",
        "uniprot_acc": "uniprot_id",
        "uniprot_id": "uniprot_id",
        "protein_id": "protein_id",
        "transcript_id": "transcript_id",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    outp = canonical_db(outp)
    col_map = {
        "hgnc_symbol": "gene_name",
        "ensembl_gene": "gene_id",
        "ensembl_transcript": "canonical_transcript",
        "ensembl_protein": "protein_id",
        "entrez_gene": "entrez_id",
        "uniprot": "uniprot_id",
    }
    out_col = col_map.get(outp)
    if out_col is None:
        return pd.DataFrame(columns=["input_id", "output_id"])

    if "query" in df.columns:
        src = df["query"].astype(str).tolist()
    elif "gene_id" in df.columns:
        src = df["gene_id"].astype(str).tolist()
    else:
        src = df.iloc[:, 0].astype(str).tolist()

    outs: list[str | None]
    if out_col in df.columns:
        outs = []
        for raw in df[out_col].astype(object).tolist():
            if raw is None:
                outs.append(None)
                continue
            raw_s = str(raw)
            if raw_s.lower() in {"nan", "none", "null"}:
                outs.append(None)
            else:
                outs.append(strip_version(raw_s))
    else:
        outs = [None] * len(src)

    return pd.DataFrame({"input_id": src, "output_id": outs})


def map_with_gget(
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
) -> pd.DataFrame:
    """Map identifiers using gget.info (Ensembl REST API-backed).

    Note: gget is Ensembl-centric, so input_db must be an Ensembl ID type.
    For non-Ensembl inputs, use 'mygene' or 'gprofiler' methods.

    Args:
        ids: Input Ensembl identifiers to map.
        input_db: Source database type. Must be one of ``"ensembl_gene"``, ``"ensembl_transcript"``,
            or ``"ensembl_protein"``.
        output_db: Target database type (e.g. ``"hgnc_symbol"``, ``"uniprot"``, ``"entrez_gene"``).
        species: Species code (e.g. ``"hsapiens"``, ``"mmusculus"``, ``"sscrofa"``).
        chunk_size: Number of IDs per API request.
        pause: Pause in seconds between requests.
        max_retries: Maximum retry attempts per chunk on failure.
        strip_versions: Strip version suffixes from Ensembl/RefSeq IDs.
        show_progress: Display progress bar.
        suppress_method_verbosity: Suppress stdout/stderr from gget.

    Returns:
        pd.DataFrame: Standardized mapping DataFrame.

    Raises:
        ValueError: If ``input_db`` is not an Ensembl type.
    """
    try:
        from gget import info as gget_info  # type: ignore
    except ImportError as e:
        raise_missing_dependency("gget", feature="gget ID mapping backend", original_error=e)

    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    # Accept multiple input ID types (Ensembl gene/transcript/protein) and
    # common gene-centric IDs that gget can resolve (HGNC symbol, UniProt, Entrez).
    if inp not in _ENSEMBL_INPUT_DB:
        allowed_str = ", ".join(sorted(_ENSEMBL_INPUT_DB))
        raise ValueError(
            f"gget input_db must be one of {{{allowed_str}}}, got {inp!r}. "
            "Tip: gget is Ensembl-centric; for other inputs try method='mygene' or 'gprofiler'."
        )

    clean_ids = [strip_version(i) if strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)
    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gget", release_used=None)

    try:
        from inspect import signature

        sig_params = set(signature(gget_info).parameters)
    except (ValueError, TypeError):
        # signature() can fail on built-in functions or unusual callables
        sig_params = set()

    kwargs: dict[str, _t.Any] = {}
    if "wrap_text" in sig_params:
        kwargs["wrap_text"] = False
    if "pdb" in sig_params:
        kwargs["pdb"] = False
    if "ensembl_only" in sig_params:
        kwargs["ensembl_only"] = False
    if "json" in sig_params:
        kwargs["json"] = False
    if "verbose" in sig_params:
        kwargs["verbose"] = False
    if "save" in sig_params:
        kwargs["save"] = False
    if "expand" in sig_params:
        kwargs["expand"] = False

    species_code = canonical_species(species)
    gget_species = {
        "hsapiens": "homo_sapiens",
        "mmusculus": "mus_musculus",
        "sscrofa": "sus_scrofa",
    }.get(species_code, species_code)
    if "species" in sig_params:
        kwargs["species"] = gget_species

    frames: list[pd.DataFrame] = []
    n_chunks = math.ceil(len(uniq_ids) / chunk_size)
    with tqdm(
        total=len(uniq_ids),
        desc="gget",
        mininterval=0.25,
        disable=not show_progress,
        ncols=100,
        unit="ids",
    ) as progress:
        for i, chunk in enumerate(_chunker(uniq_ids, chunk_size), start=1):
            logger.debug(f"gget: querying chunk {i}/{n_chunks} (n={len(chunk)})")
            attempt = 0
            while True:
                attempt += 1
                try:
                    with _suppress_stdout_stderr(suppress_method_verbosity):
                        df_raw = gget_info(chunk, **kwargs)
                    if not isinstance(df_raw, pd.DataFrame):
                        df_raw = pd.DataFrame(df_raw)
                    part = _gget_extract(df_raw, outp)
                    frames.append(part)
                    break
                except Exception as e:
                    logger.warning(f"gget batch failed (attempt {attempt}): {e}")
                    if attempt >= max_retries:
                        frames.append(
                            pd.DataFrame(
                                {
                                    "input_id": chunk,
                                    "output_id": [None] * len(chunk),
                                }
                            )
                        )
                        break
                    time.sleep(pause * attempt)
            progress.update(len(chunk))
            time.sleep(pause)

    if not frames:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "gget", release_used=None)

    out = pd.concat(frames, ignore_index=True)

    if "input_id" not in out.columns:
        out["input_id"] = None
    out["input_id"] = out["input_id"].astype(str)
    if strip_versions:
        out["input_id"] = out["input_id"].apply(strip_version)
    else:
        out["input_id"] = out["input_id"].str.strip()

    out["input_db"] = inp
    out["output_db"] = outp
    out["method"] = "gget"
    out["release_used"] = None
    out["metadata_json"] = _json({})
    out = _ensure_all_inputs(out, clean_ids, inp, outp, "gget", release_used=None)
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
