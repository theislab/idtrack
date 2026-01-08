#!/usr/bin/env python3
"""Ensembl BioMart backend for ID mapping.

This module provides the map_with_pybiomart() function for querying
Ensembl BioMart to convert biological identifiers. Supports historical
Ensembl releases via archive hosts.
"""

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

import math
import re
import time
import typing as _t
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from idtrack._external_mappers._constants import (
    _BM_ATTR_CANDIDATES,
    _BM_FILTER_CANDIDATES,
    _ENSEMBL_ARCHIVE_BY_RELEASE,
    _ENSEMBL_INPUT_DB,
    _ENSEMBL_SPECIAL_RELEASE_HOSTS,
)
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


def _ensembl_archive_host_for_release(
    release: int | str | None,
) -> str | None:
    """Resolve an Ensembl release or key to an archive host.

    Examples include an integer release (e.g. ``104``) or a special string key
    (e.g. ``"GRCh37"``) mapping to hosts like ``"may2021.archive.ensembl.org"``.

    Args:
        release: Ensembl release number or key (e.g. ``104``, ``"v104"``, ``"GRCh37"``), or ``None``.

    Returns:
        Archive host for the requested release, or ``None`` if unknown.
    """
    if release is None:
        return None

    # String releases can be things like "104", "v104", "GRCh37", ...
    if isinstance(release, str):
        s = release.strip()
        if not s:
            return None

        key = s.lower()
        # Special non‑numeric keys first
        if key in _ENSEMBL_SPECIAL_RELEASE_HOSTS:
            return _ENSEMBL_SPECIAL_RELEASE_HOSTS[key]

        # Strip a leading "v" or "r" if present (e.g. "v104")
        m = re.match(r"^[vr]?(\d+)$", key)
        if not m:
            return None
        try:
            rel_int = int(m.group(1))
        except ValueError:
            return None
    else:
        try:
            rel_int = int(release)
        except (TypeError, ValueError):
            return None

    return _ENSEMBL_ARCHIVE_BY_RELEASE.get(rel_int)


def _biomart_dataset_for_species(species: str, explicit: str | None = None) -> str:
    """Return the Ensembl BioMart dataset name for the given species."""
    if explicit:
        return explicit
    s = canonical_species(species)
    return f"{s}_gene_ensembl"


def _normalize_biomart_host(host: str | None) -> str:
    """Normalize an Ensembl BioMart host for pybiomart.

    Examples of valid outputs:
        "http://www.ensembl.org"
        "http://nov2020.archive.ensembl.org"
        "http://grch37.ensembl.org"

    Args:
        host: Hostname or URL (scheme optional). If ``None``, defaults to ``"http://www.ensembl.org"``.

    Returns:
        str: Normalized base URL suitable for pybiomart.
    """
    if not host:
        return "http://www.ensembl.org"

    host = host.strip()
    parsed = urlparse(host if "://" in host else "http://" + host)
    netloc = parsed.netloc or parsed.path.split("/")[0]
    return "http://" + netloc


def _bm_list_attribute_names(ds) -> list[str]:
    """Return a list of attribute names for a pybiomart Dataset."""
    try:
        attrs = ds.list_attributes()
    except (AttributeError, TypeError, RuntimeError):
        # Different pybiomart versions may not have list_attributes()
        attrs = getattr(ds, "attributes", None)

    if attrs is None:
        return []

    try:
        # biomaRt-style DataFrame
        if hasattr(attrs, "columns"):
            if "name" in attrs.columns:
                return [str(x) for x in attrs["name"].tolist()]
            else:
                return [str(x) for x in attrs.iloc[:, 0].tolist()]
        # dict or list-like
        if isinstance(attrs, dict):
            return [str(k) for k in attrs.keys()]
        return [str(x) for x in list(attrs)]
    except (KeyError, IndexError, TypeError, AttributeError):
        try:
            return list(attrs)
        except (TypeError, ValueError):
            return []


def _bm_list_filter_names(ds) -> list[str]:
    """Return a list of filter names for a pybiomart Dataset."""
    try:
        filts = ds.list_filters()
    except (AttributeError, TypeError, RuntimeError):
        # Different pybiomart versions may not have list_filters()
        filts = getattr(ds, "filters", None)

    if filts is None:
        return []

    try:
        if hasattr(filts, "columns"):
            if "name" in filts.columns:
                return [str(x) for x in filts["name"].tolist()]
            else:
                return [str(x) for x in filts.iloc[:, 0].tolist()]
        if isinstance(filts, dict):
            return [str(k) for k in filts.keys()]
        return [str(x) for x in list(filts)]
    except (KeyError, IndexError, TypeError, AttributeError):
        try:
            return list(filts)
        except (TypeError, ValueError):
            return []


def _bm_pick_attribute(canonical_db_name: str, available_attrs: list[str]) -> str:
    """Choose a BioMart attribute name for a canonical DB.

    The helper first tries explicit candidates from :py:data:`_BM_ATTR_CANDIDATES` and
    falls back to fuzzy matching on common substrings.

    Args:
        canonical_db_name: Canonical database key (see :py:func:`~idtrack._external_mappers._utils.canonical_db`).
        available_attrs: Attribute names provided by the BioMart dataset.

    Returns:
        str: Selected attribute name.

    Raises:
        RuntimeError: If no compatible attribute is available on the dataset.
    """
    cdb = canonical_db(canonical_db_name)
    attrs = list(dict.fromkeys(available_attrs))  # dedupe, preserve order
    attr_set = set(attrs)

    # 1) Try explicit candidates in preferred order
    candidates = _BM_ATTR_CANDIDATES.get(cdb, [])
    for cand in candidates:
        if cand in attr_set:
            return cand

    # 2) Fuzzy fallback based on typical naming patterns
    if cdb.startswith("ensembl_"):
        pattern = "ensembl_" + cdb.split("_", 1)[1]
    elif cdb == "hgnc_symbol":
        pattern = "external_gene"
    elif cdb == "hgnc_id":
        pattern = "hgnc"
    elif cdb == "entrez_gene":
        pattern = "entrez"
    elif cdb == "uniprot":
        pattern = "uniprot"
    elif cdb.startswith("refseq_"):
        pattern = "refseq"
    else:
        pattern = cdb

    prefix_hits = [a for a in attrs if a.startswith(pattern)]
    if prefix_hits:
        return prefix_hits[0]

    contains_hits = [a for a in attrs if pattern in a]
    if contains_hits:
        return contains_hits[0]

    raise RuntimeError(
        f"pybiomart: dataset does not provide any attribute compatible with "
        f"{cdb!r}; inspect `dataset.list_attributes()` for valid names."
    )


def _bm_pick_filter(
    canonical_db_name: str,
    attr_name: str,
    available_filters: list[str],
) -> str:
    """Choose a BioMart filter name.

    The selection depends on the canonical database and chosen attribute.

    Args:
        canonical_db_name: Canonical database key for the input IDs.
        attr_name: Attribute name chosen for the input IDs.
        available_filters: Filter names provided by the BioMart dataset.

    Returns:
        str: Selected filter name.

    Raises:
        RuntimeError: If no compatible filter is available on the dataset.
    """
    cdb = canonical_db(canonical_db_name)
    filt_list = list(dict.fromkeys(available_filters))  # dedupe, preserve order
    filt_set = set(filt_list)

    candidates: list[str] = []

    # 1) DB-specific candidates
    candidates.extend(_BM_FILTER_CANDIDATES.get(cdb, []))

    # 2) The attribute name itself
    candidates.append(attr_name)

    # 3) Some small variations
    if attr_name.endswith("_id"):
        candidates.append(attr_name[:-3])
    if attr_name.endswith("_accession"):
        candidates.append(attr_name[:-10])

    # Deduplicate candidates but preserve order
    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered_candidates.append(c)

    for c in ordered_candidates:
        if c in filt_set:
            return c

    # 4) Fuzzy search
    if cdb == "hgnc_symbol":
        pattern = "external_gene"
    elif cdb.startswith("ensembl_"):
        pattern = "ensembl_" + cdb.split("_", 1)[1]
    elif cdb == "entrez_gene":
        pattern = "entrez"
    elif cdb == "uniprot":
        pattern = "uniprot"
    elif cdb.startswith("refseq_"):
        pattern = "refseq"
    else:
        pattern = cdb

    prefix_hits = [f for f in filt_list if f.startswith(pattern)]
    if prefix_hits:
        return prefix_hits[0]

    contains_hits = [f for f in filt_list if pattern in f]
    if contains_hits:
        return contains_hits[0]

    raise RuntimeError(
        f"pybiomart: dataset does not provide any filter compatible with "
        f"{cdb!r}; inspect `dataset.list_filters()` for valid names."
    )


def map_with_pybiomart(
    ids: _t.Iterable[str],
    input_db: str,
    output_db: str,
    *,
    species: str = "hsapiens",
    chunk_size: int = 1000,
    pause: float = 0.2,
    strip_versions: bool = True,
    release: str | int | None = None,
    show_progress: bool = True,
    suppress_method_verbosity: bool = True,
) -> pd.DataFrame:
    """Map identifiers using Ensembl BioMart via pybiomart.

    Note: BioMart can only filter by Ensembl IDs (gene, transcript, protein).
    Other ID types can be used as output_db but not input_db.

    Args:
        ids: Input Ensembl identifiers to map.
        input_db: Source database type. Must be one of ``"ensembl_gene"``, ``"ensembl_transcript"``,
            or ``"ensembl_protein"``.
        output_db: Target database type (e.g. ``"hgnc_symbol"``, ``"uniprot"``, ``"entrez_gene"``).
        species: Species code (e.g. ``"hsapiens"``, ``"mmusculus"``, ``"sscrofa"``).
        chunk_size: Number of IDs per BioMart query.
        pause: Pause in seconds between queries.
        strip_versions: Strip version suffixes from Ensembl/RefSeq IDs.
        release: Ensembl release number (e.g. ``104``) or special key (e.g. ``"grch37"``). If ``None``, uses
            the current Ensembl release.
        show_progress: Display progress bar.
        suppress_method_verbosity: Suppress stdout/stderr from pybiomart.

    Returns:
        pd.DataFrame: Standardized mapping DataFrame.

    Raises:
        RuntimeError: If the BioMart connection fails or required dataset metadata cannot be retrieved.
        ValueError: If ``input_db`` is not an Ensembl type.
    """
    try:
        from pybiomart import Dataset  # type: ignore
    except ImportError as e:
        raise_missing_dependency("pybiomart", feature="pybiomart ID mapping backend", original_error=e)

    inp = canonical_db(input_db)
    outp = canonical_db(output_db)

    # Early, explicit check: BioMart can only *filter* by Ensembl IDs.
    # It can still *return* HGNC/UniProt/etc. as attributes, but input_db
    # must be one of the Ensembl IDs.
    if inp not in _ENSEMBL_INPUT_DB:
        allowed_str = ", ".join(sorted(_ENSEMBL_INPUT_DB))
        raise ValueError(
            f"pybiomart input_db must be one of {{{allowed_str}}}, got {inp!r}. "
            "BioMart cannot filter by HGNC/UniProt directly; keep them as output_db "
            "or use method='mygene'/'gprofiler' for those inputs."
        )

    clean_ids = [strip_version(i) if strip_versions else str(i) for i in ids]
    uniq_ids = _unique_not_null(clean_ids)

    if not uniq_ids:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "pybiomart", release_used=None)

    # Resolve Ensembl archive host solely from `release` (if provided).
    raw_host: str | None = None

    resolved_release_host: str | None = None

    if release is not None:
        resolved_release_host = _ensembl_archive_host_for_release(release)
        if resolved_release_host:
            raw_host = resolved_release_host
        else:
            logger.warning(
                "pybiomart: no known archive host for Ensembl release %r; falling back to www.ensembl.org",
                release,
            )

    if raw_host is None:
        raw_host = "http://www.ensembl.org"

    host = _normalize_biomart_host(raw_host)

    dataset_name = _biomart_dataset_for_species(species)

    try:
        ds = Dataset(name=dataset_name, host=host)
    except Exception as e:
        raise RuntimeError(
            f"pybiomart: failed to connect to Ensembl BioMart " f"(dataset={dataset_name!r}, host={host!r}): {e}"
        ) from e

    # Discover attributes and filters that actually exist for this dataset
    attr_names = _bm_list_attribute_names(ds)
    filter_names = _bm_list_filter_names(ds)

    if not attr_names:
        raise RuntimeError(f"pybiomart: could not retrieve attributes for dataset {dataset_name!r}")
    if not filter_names:
        raise RuntimeError(f"pybiomart: could not retrieve filters for dataset {dataset_name!r}")

    # Choose valid attribute + filter names for the requested mapping
    in_attr = _bm_pick_attribute(inp, attr_names)
    out_attr = _bm_pick_attribute(outp, attr_names)
    filter_name = _bm_pick_filter(inp, in_attr, filter_names)

    logger.debug(
        "pybiomart: using dataset=%r host=%r in_attr=%r out_attr=%r filter=%r",
        dataset_name,
        host,
        in_attr,
        out_attr,
        filter_name,
    )

    frames: list[pd.DataFrame] = []
    n_chunks = math.ceil(len(uniq_ids) / chunk_size)
    with tqdm(
        total=len(uniq_ids),
        desc="pybiomart",
        mininterval=0.25,
        disable=not show_progress,
        ncols=100,
        unit="ids",
    ) as progress:
        for i, chunk in enumerate(_chunker(uniq_ids, chunk_size), start=1):
            logger.debug(
                "pybiomart: querying chunk %d/%d (n=%d)",
                i,
                n_chunks,
                len(chunk),
            )
            try:
                with _suppress_stdout_stderr(suppress_method_verbosity):
                    df = ds.query(
                        attributes=[in_attr, out_attr],
                        filters={filter_name: chunk},
                        use_attr_names=True,
                    )

                if df is None or df.empty:
                    frames.append(
                        pd.DataFrame(
                            {
                                "input_id": chunk,
                                "output_id": [None] * len(chunk),
                            }
                        )
                    )
                else:
                    df = df.rename(columns={in_attr: "input_id", out_attr: "output_id"})
                    keep_cols = [c for c in ("input_id", "output_id") if c in df.columns]
                    if not keep_cols:
                        frames.append(
                            pd.DataFrame(
                                {
                                    "input_id": chunk,
                                    "output_id": [None] * len(chunk),
                                }
                            )
                        )
                    else:
                        frames.append(df[keep_cols].drop_duplicates())

            except Exception as e:
                logger.warning(f"pybiomart chunk failed: {e}")
                meta = {"error": str(e)}
                frames.append(
                    pd.DataFrame(
                        {
                            "input_id": chunk,
                            "output_id": [None] * len(chunk),
                            "metadata_json": [_json(meta)] * len(chunk),
                        }
                    )
                )

            progress.update(len(chunk))
            time.sleep(pause)

    if not frames:
        return _ensure_all_inputs(_empty_result(), clean_ids, inp, outp, "pybiomart", release_used=host)

    out = pd.concat(frames, ignore_index=True)

    out["input_db"] = inp
    out["output_db"] = outp
    out["method"] = "pybiomart"
    out["release_used"] = host
    if "metadata_json" not in out.columns:
        out["metadata_json"] = _json({})

    out = _ensure_all_inputs(out, clean_ids, inp, outp, "pybiomart", release_used=host)
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
