#!/usr/bin/env python3
"""
Utility functions for the _external_mappers module.

This module provides:
- Database and species name canonicalization
- ID version stripping for Ensembl and RefSeq identifiers
- DataFrame utilities for standardizing output format
- Helper functions for chunking, JSON serialization, etc.
"""

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import typing
from typing import Any, Iterable

import pandas as pd

from idtrack._external_mappers._constants import _DB_ALIASES, _SPECIES_ALIASES, SUPPORTED_DBS

__all__ = [
    "canonical_db",
    "canonical_species",
    "_species_for_mygene",
    "strip_version",
    "_as_list",
    "_unique_not_null",
    "_chunker",
    "_json",
    "_is_bare_numeric",
    "_empty_result",
    "_add_mapping_column",
    "_ensure_all_inputs",
    "_suppress_stdout_stderr",
    "logger",
    "check_optional_dependencies",
    "raise_missing_dependency",
]


# ----------------------------- Dependency Info ------------------------------ #

# Registry of optional dependencies for this module
OPTIONAL_DEPENDENCIES: dict[str, dict[str, Any]] = {
    "gget": {
        "import_name": "gget",
        "pip_name": "gget",
        "features": ["gget backend", "ortholog utilities"],
        "description": "Query Ensembl REST API for gene information",
    },
    "mygene": {
        "import_name": "mygene",
        "pip_name": "mygene",
        "features": ["mygene backend"],
        "description": "Query MyGene.info API for ID mapping",
    },
    "pybiomart": {
        "import_name": "pybiomart",
        "pip_name": "pybiomart",
        "features": ["pybiomart backend"],
        "description": "Query Ensembl BioMart for ID mapping",
    },
    "gprofiler-official": {
        "import_name": "gprofiler",
        "pip_name": "gprofiler-official",
        "features": ["gprofiler backend"],
        "description": "Query g:Profiler API for ID mapping",
    },
    "biopython": {
        "import_name": "Bio",
        "pip_name": "biopython",
        "features": ["ortholog utilities", "sequence alignment"],
        "description": "Biological sequence analysis tools",
    },
}


def _check_dependency(dep_key: str) -> bool:
    """Check if a single dependency is available."""
    if dep_key not in OPTIONAL_DEPENDENCIES:
        return False
    import_name = OPTIONAL_DEPENDENCIES[dep_key]["import_name"]
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_optional_dependencies(warn: bool = True) -> dict[str, bool]:
    """
    Check which optional dependencies are installed.

    Parameters
    ----------
    warn : bool, default True
        If True, emit a warning for each missing dependency.

    Returns
    -------
    dict[str, bool]
        Dictionary mapping dependency names to availability status.
    """
    import warnings

    status: dict[str, bool] = {}
    missing: list[str] = []

    for dep_key in OPTIONAL_DEPENDENCIES:
        available = _check_dependency(dep_key)
        status[dep_key] = available
        if not available:
            missing.append(dep_key)

    if warn and missing:
        pip_names = [OPTIONAL_DEPENDENCIES[d]["pip_name"] for d in missing]
        warning_msg = (
            f"\n"
            f"{'=' * 70}\n"
            f"idtrack._external_mappers: Missing optional dependencies\n"
            f"{'=' * 70}\n"
            f"\n"
            f"The following optional packages are not installed:\n"
        )
        for dep_key in missing:
            info = OPTIONAL_DEPENDENCIES[dep_key]
            features = ", ".join(info["features"])
            warning_msg += f"  - {info['pip_name']}: {info['description']} (used by: {features})\n"

        warning_msg += (
            f"\n"
            f"To install all external mapper dependencies:\n"
            f"  pip install {' '.join(pip_names)}\n"
            f"\n"
            f"Or install individually as needed.\n"
            f"{'=' * 70}\n"
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)

    return status


def raise_missing_dependency(
    dep_key: str,
    feature: str | None = None,
    original_error: BaseException | None = None,
) -> typing.NoReturn:
    """
    Raise a detailed error for a missing optional dependency.

    Parameters
    ----------
    dep_key : str
        Key in OPTIONAL_DEPENDENCIES (e.g., "gget", "mygene").
    feature : str, optional
        Description of the feature that requires this dependency.
    original_error : BaseException, optional
        The original ImportError to chain.

    Raises
    ------
    RuntimeError
        Always raises with detailed installation instructions.
    """
    if dep_key not in OPTIONAL_DEPENDENCIES:
        # Fallback for unknown dependencies
        msg = (
            f"\n"
            f"{'=' * 70}\n"
            f"Missing dependency: {dep_key}\n"
            f"{'=' * 70}\n"
            f"\n"
            f"Please install the required package:\n"
            f"  pip install {dep_key}\n"
            f"{'=' * 70}\n"
        )
        raise RuntimeError(msg) from original_error

    info = OPTIONAL_DEPENDENCIES[dep_key]
    pip_name = info["pip_name"]
    description = info["description"]
    features = ", ".join(info["features"])

    if feature is None:
        feature = features

    msg = (
        f"\n"
        f"{'=' * 70}\n"
        f"Missing optional dependency: {pip_name}\n"
        f"{'=' * 70}\n"
        f"\n"
        f"The '{feature}' feature requires '{pip_name}'.\n"
        f"\n"
        f"Package info:\n"
        f"  - Name: {pip_name}\n"
        f"  - Description: {description}\n"
        f"  - Used by: {features}\n"
        f"\n"
        f"To install this dependency:\n"
        f"  pip install {pip_name}\n"
        f"\n"
        f"To install all external mapper dependencies:\n"
        f"  pip install gget mygene pybiomart gprofiler-official biopython\n"
        f"{'=' * 70}\n"
    )
    raise RuntimeError(msg) from original_error


# ------------------------------- Logging ------------------------------------ #


def _setup_logger() -> logging.Logger:
    """Set up and return the module logger with proper configuration."""
    log = logging.getLogger("idtrack.external_mappers")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


logger = _setup_logger()


@contextlib.contextmanager
def _suppress_stdout_stderr(enabled: bool):
    """Context manager to squelch noisy stdout/stderr emissions."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ----------------------------- Canonical DBs -------------------------------- #


def canonical_db(db: str) -> str:
    """Return canonical DB key given a user-friendly/alias string."""
    if not isinstance(db, str) or not db.strip():
        raise ValueError("db must be a non-empty string")
    db_norm = db.strip().lower()
    if db_norm in _DB_ALIASES:
        return _DB_ALIASES[db_norm]
    if db_norm.startswith("ensg"):
        return "ensembl_gene"
    if db_norm.startswith("enst"):
        return "ensembl_transcript"
    if db_norm.startswith("ensp"):
        return "ensembl_protein"
    raise ValueError(f"Unsupported or unknown db alias: {db!r}. " f"Supported canonical DBs: {sorted(SUPPORTED_DBS)}")


# -------------------------- Species normalization --------------------------- #


def canonical_species(species: str | None) -> str:
    """
    Canonical organism code (g:Profiler / Ensembl style).

    Supported out-of-the-box:
        - human: hsapiens
        - mouse: mmusculus
        - pig:   sscrofa
    """
    if not species:
        return "hsapiens"
    s = species.strip().lower()
    return _SPECIES_ALIASES.get(s, s)


def _species_for_mygene(species: str | None) -> str:
    """MyGene expects common names like 'human' / 'mouse' / 'pig'."""
    cs = canonical_species(species)
    if cs == "hsapiens":
        return "human"
    if cs == "mmusculus":
        return "mouse"
    if cs == "sscrofa":
        return "pig"
    return cs


# ----------------------------- Helper utils -------------------------------- #

# Regex patterns for ID version stripping
_ENS_RE = re.compile(r"^(ENS[A-Z]*\d+)")
_REFSEQ_VER_RE = re.compile(r"^([NX][MRP]_\d+)")


def strip_version(id_: str) -> str:
    """
    Strip version suffixes from Ensembl and RefSeq identifiers.

    Examples:
        >>> strip_version("ENSG00000141510.15")
        'ENSG00000141510'
        >>> strip_version("NM_000546.6")
        'NM_000546'
        >>> strip_version("TP53")  # Non-versioned IDs pass through
        'TP53'

    Parameters
    ----------
    id_ : str
        The identifier to strip version from.

    Returns
    -------
    str
        The identifier without version suffix.
    """
    if not isinstance(id_, str):
        return id_  # type: ignore[return-value]
    x = id_.strip()

    m = _ENS_RE.match(x)
    if m:
        return m.group(1)

    m = _REFSEQ_VER_RE.match(x)
    if m:
        return m.group(1)

    return x


def _as_list(v: Any) -> list[Any]:
    """
    Coerce a value to a list.

    - None returns []
    - list/tuple/set returns list(v)
    - Scalar values return [v]
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _unique_not_null(seq: Iterable[Any]) -> list[str]:
    """
    Return unique non-null string values from a sequence, preserving order.

    Filters out:
    - None values
    - Empty strings or whitespace-only strings
    - String representations of null values: 'nan', 'none', 'null' (case-insensitive)

    Parameters
    ----------
    seq : Iterable[Any]
        Input sequence of values.

    Returns
    -------
    list[str]
        List of unique non-null string values, preserving first occurrence order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for v in seq:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _chunker(items: list[Any], size: int) -> typing.Iterator[list[Any]]:
    """
    Yield successive chunks of a given size from a list.

    Parameters
    ----------
    items : list
        The list to chunk.
    size : int
        Maximum size of each chunk.

    Yields
    ------
    list
        Successive chunks of the input list.
    """
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _json(obj: Any) -> str:
    """
    Serialize object to compact JSON string with unicode preserved.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    str
        Compact JSON string representation.
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _is_bare_numeric(s: str) -> bool:
    """
    Check if a string consists entirely of digits.

    Parameters
    ----------
    s : str
        String to check.

    Returns
    -------
    bool
        True if the string is a bare numeric (all digits), False otherwise.
    """
    return bool(re.fullmatch(r"\d+", str(s).strip()))


# ---------------------------- Utilities/Finalizers -------------------------- #


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        ]
    )


def _add_mapping_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add / recompute the `mapping` column on a DataFrame with at least
    `input_id` and `output_id` columns.

    mapping is per-input cardinality of the mapping:
        1:0  -> no non-null output_id
        1:1  -> exactly one unique non-null output_id
        1:n  -> more than one unique non-null output_id
    """
    if df is None:
        return _empty_result()
    if df.empty:
        if "mapping" not in df.columns:
            df["mapping"] = pd.Series(dtype=object)
        return df

    if "output_id" not in df.columns or "input_id" not in df.columns:
        # Fallback: mark everything as unmapped
        df["mapping"] = "1:0"
        return df

    # Drop any existing mapping column; we'll recompute it from scratch.
    if "mapping" in df.columns:
        df = df.drop(columns=["mapping"])

    inputs = df["input_id"].astype(str)
    outputs = df["output_id"]

    out_str = outputs.astype(str)
    valid_mask = ~outputs.isna() & out_str.str.strip().ne("") & ~out_str.str.lower().isin({"nan", "none", "null"})

    # Number of unique valid outputs per input
    counts_by_input = df[valid_mask].groupby(inputs[valid_mask])["output_id"].nunique(dropna=True)

    mapping_by_input: dict[str, str] = {}
    for inp_val in inputs.unique():
        key = str(inp_val)
        n = int(counts_by_input.get(key, 0) or 0)
        if n == 0:
            mapping_by_input[key] = "1:0"
        elif n == 1:
            mapping_by_input[key] = "1:1"
        else:
            mapping_by_input[key] = "1:n"

    df["mapping"] = inputs.map(mapping_by_input)

    return df


def _ensure_all_inputs(
    df: pd.DataFrame,
    original_inputs: list[str],
    inp: str,
    outp: str,
    method: str,
    release_used: str | None,
) -> pd.DataFrame:
    """
    Guarantee that each input appears at least once in the output
    (with output_id=None if unmapped). Preserve input order and add
    the `mapping` column.
    """
    if df is None or df.empty:
        base = pd.DataFrame(
            {
                "input_id": [str(x) for x in original_inputs],
                "input_db": inp,
                "output_id": [None] * len(original_inputs),
                "output_db": outp,
                "method": method,
                "release_used": release_used,
                "metadata_json": _json({}),
            }
        )
        base = _add_mapping_column(base)
        return base

    for col in (
        "input_id",
        "input_db",
        "output_id",
        "output_db",
        "method",
        "release_used",
        "metadata_json",
    ):
        if col not in df.columns:
            if col in {"input_db", "output_db", "method"}:
                default_map = {"input_db": inp, "output_db": outp, "method": method}
                df[col] = default_map[col]
            elif col == "release_used":
                df[col] = release_used
            elif col == "metadata_json":
                df[col] = _json({})
            else:
                df[col] = None

    present = set(df["input_id"].astype(str))
    missing = [x for x in original_inputs if str(x) not in present]
    if missing:
        tail = pd.DataFrame(
            {
                "input_id": missing,
                "input_db": inp,
                "output_id": [None] * len(missing),
                "output_db": outp,
                "method": method,
                "release_used": release_used,
                "metadata_json": _json({}),
            }
        )
        df = pd.concat([df, tail], ignore_index=True)

    order_map = {str(x): i for i, x in enumerate(original_inputs)}
    df = df.copy()
    df["__ord"] = df["input_id"].astype(str).map(order_map)
    df = df.sort_values(["__ord", "output_id"], na_position="last").drop(columns="__ord")
    df = df.reset_index(drop=True)

    df = _add_mapping_column(df)
    return df
