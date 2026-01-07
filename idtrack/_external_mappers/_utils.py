#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com


from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import typing

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
]


# ------------------------------- Logging ------------------------------------ #

logger = logging.getLogger("id_mapper")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


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

_ENS_RE = re.compile(r"^(ENS[A-Z]*\d+)")
_REFSEQ_VER_RE = re.compile(r"^([NX][MRP]_\d+)")
_WS = re.compile(r"\s+")


def strip_version(id_: str) -> str:
    """Strip version suffixes for Ensembl/RefSeq when appropriate."""
    if not isinstance(id_, str):
        return id_
    x = id_.strip()
    
    m = _ENS_RE.match(x)
    if m:
        return m.group(1)
    
    m = _REFSEQ_VER_RE.match(x)
    if m:
        return m.group(1)
    
    return x

def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    return [v]


def _unique_not_null(seq: typing.Iterable[typing.Any]) -> list[str]:
    seen, out = set(), []
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


def _chunker(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _is_bare_numeric(s: str) -> bool:
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
