"""External-mapper evaluation helpers for experiment notebooks.

This module is intentionally graph-free: it only uses `idtrack._external_mappers`
and operates on the standardized DataFrame outputs it returns.

Primary use cases (marketing + robustness):
  - Run multiple external mapper backends in a uniform way
  - Gracefully skip methods when optional dependencies are missing
  - Normalize outputs for downstream set-based agreement metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

import idtrack._external_mappers as ext
from idtrack._external_mappers._utils import check_optional_dependencies

__all__ = [
    "ExternalRun",
    "available_external_methods",
    "normalize_external_df",
    "run_external_methods",
]


_METHOD_TO_DEP_KEY: dict[str, str] = {
    "pybiomart": "pybiomart",
    "mygene": "mygene",
    "gprofiler": "gprofiler-official",
    "gget": "gget",
}


@dataclass(frozen=True)
class ExternalRun:
    """A single external-mapper run specification."""

    ids: list[str]
    input_db: str
    output_db: str
    species: str = "hsapiens"
    methods: tuple[str, ...] = ("pybiomart", "mygene", "gprofiler", "gget")
    pybiomart_release: str | int | None = None
    chunk_size: int = 200
    pause: float = 0.1
    max_retries: int = 3
    strip_versions: bool = True
    verbose: int | str | bool = 2
    suppress_method_verbosity: bool = True


def available_external_methods(methods: Iterable[str], *, warn: bool = False) -> dict[str, bool]:
    """Return a method->available mapping based on installed optional dependencies."""

    dep_status = check_optional_dependencies(warn=warn)
    out: dict[str, bool] = {}
    for m in methods:
        key = str(m).strip().lower()
        dep_key = _METHOD_TO_DEP_KEY.get(key)
        if dep_key is None:
            # Unknown method; let the caller handle.
            out[key] = False
            continue
        out[key] = bool(dep_status.get(dep_key, False))
    return out


def normalize_external_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize an external-mapper DataFrame to a stable column subset."""

    if df is None or df.empty:
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

    cols = [
        c
        for c in [
            "input_id",
            "input_db",
            "output_id",
            "output_db",
            "method",
            "release_used",
            "mapping",
            "metadata_json",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()

    # Ensure 'mapping' exists (older cached frames may omit it).
    if "mapping" not in out.columns:
        per = (
            out.assign(output_id=out["output_id"].astype("object"))
            .groupby("input_id", as_index=False)["output_id"]
            .nunique(dropna=True)
            .rename(columns={"output_id": "n_outputs"})
        )
        out = out.merge(per, on="input_id", how="left")
        out["mapping"] = out["n_outputs"].map(lambda n: "1:0" if not n else ("1:1" if int(n) == 1 else "1:n"))
        out.drop(columns=["n_outputs"], inplace=True)

    return out.reset_index(drop=True)


def run_external_methods(run: ExternalRun) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Run external mapper backends and return (results, errors)."""

    ids = [str(x) for x in (run.ids or [])]
    methods = [str(m).strip().lower() for m in (run.methods or ())]

    available = available_external_methods(methods, warn=False)

    results: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, Any]] = []

    for method in methods:
        if method not in ext.SUPPORTED_METHODS:
            errors.append(
                {
                    "method": method,
                    "error": f"Unsupported method {method!r}. Supported: {ext.SUPPORTED_METHODS}",
                }
            )
            results[method] = normalize_external_df(None)
            continue

        if not available.get(method, False):
            dep_key = _METHOD_TO_DEP_KEY.get(method, "unknown")
            errors.append(
                {
                    "method": method,
                    "error": f"Missing optional dependency for {method!r} (dep_key={dep_key!r}).",
                }
            )
            results[method] = normalize_external_df(None)
            continue

        kwargs: dict[str, Any] = {
            "ids": ids,
            "input_db": run.input_db,
            "output_db": run.output_db,
            "method": method,
            "species": run.species,
            "chunk_size": int(run.chunk_size),
            "pause": float(run.pause),
            "max_retries": int(run.max_retries),
            "strip_versions": bool(run.strip_versions),
            "verbose": run.verbose,
            "suppress_method_verbosity": bool(run.suppress_method_verbosity),
        }
        if method == "pybiomart":
            kwargs["release_for_pybiomart"] = run.pybiomart_release

        try:
            df = ext.convert_ids(**kwargs)
            results[method] = normalize_external_df(df)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "method": method,
                    "error": str(exc),
                    "kwargs": {k: v for k, v in kwargs.items() if k not in {"ids"}},
                    "n_ids": len(ids),
                }
            )
            results[method] = normalize_external_df(None)

    return results, errors

