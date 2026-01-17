"""Time-travel (release×release) analysis helpers.

This module is intentionally lightweight:
  - consumes cached DataFrames produced by experiment notebooks
  - does not import/require IDTrack graph loading
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "add_fraction_columns",
    "aggregate_bootstraps",
    "asymmetry_matrix",
    "directional_distance_curve",
    "release_stability_index",
]


def add_fraction_columns(df: pd.DataFrame, *, denom_col: str = "total") -> pd.DataFrame:
    """Add standard fraction columns used across time-travel notebooks."""

    if df.empty:
        return df.copy()

    out = df.copy()
    den = out[denom_col].replace(0, np.nan)

    # Core outcome fractions (IDTrack results schema via summarize_binned_conversion)
    mapping = {
        "frac_1_to_0": "1_to_0",
        "frac_1_to_1_tdm": "1_to_1_tdm",
        "frac_1_to_1_atm": "1_to_1_atm",
        "frac_1_to_n_tdm": "1_to_n_tdm",
        "frac_1_to_n_atm": "1_to_n_atm",
        "frac_1_to_1_total": "1_to_1_total",
        "frac_1_to_n_total": "1_to_n_total",
        "frac_changed_1_to_1": "changed_1_to_1",
        "frac_changed_1_to_n": "changed_1_to_n",
    }
    for frac, raw in mapping.items():
        if raw in out.columns and frac not in out.columns:
            out[frac] = out[raw] / den

    if "frac_atm_total" not in out.columns:
        if "1_to_1_atm" in out.columns and "1_to_n_atm" in out.columns:
            out["frac_atm_total"] = (out["1_to_1_atm"] + out["1_to_n_atm"]) / den

    if "frac_tdm_total" not in out.columns:
        if "1_to_1_tdm" in out.columns and "1_to_n_tdm" in out.columns:
            out["frac_tdm_total"] = (out["1_to_1_tdm"] + out["1_to_n_tdm"]) / den

    if "frac_changed_any" not in out.columns:
        if "changed_1_to_1" in out.columns and "changed_1_to_n" in out.columns:
            out["frac_changed_any"] = (out["changed_1_to_1"] + out["changed_1_to_n"]) / den

    return out


def aggregate_bootstraps(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate a bootstrap-expanded grid into mean/std columns."""

    if df.empty:
        return df.copy()

    numeric_cols = [c for c in df.columns if c not in set(group_cols)]
    mean = df.groupby(group_cols, as_index=False)[numeric_cols].mean(numeric_only=True)
    std = (
        df.groupby(group_cols, as_index=False)[numeric_cols]
        .std(numeric_only=True)
        .rename(columns={c: f"std_{c}" for c in numeric_cols})
    )
    return mean.merge(std, on=group_cols, how="left")


def asymmetry_matrix(
    agg: pd.DataFrame,
    *,
    metric: str,
    final_database: str,
    from_col: str = "from_release",
    to_col: str = "to_release",
    final_db_col: str = "final_database",
) -> pd.DataFrame:
    """Compute a signed asymmetry matrix: metric(A→B) - metric(B→A)."""

    sub = agg[agg[final_db_col] == final_database].copy()
    if sub.empty:
        return pd.DataFrame()
    if metric not in sub.columns:
        raise KeyError(f"metric {metric!r} not present in agg columns")

    fwd = sub[[from_col, to_col, metric]].copy()
    rev = fwd.rename(columns={from_col: to_col, to_col: from_col, metric: "metric_rev"})
    m = fwd.merge(rev, on=[from_col, to_col], how="inner")
    m["asymmetry"] = m[metric] - m["metric_rev"]
    mat = m.pivot(index=from_col, columns=to_col, values="asymmetry").sort_index().sort_index(axis=1)
    return mat


def directional_distance_curve(
    agg: pd.DataFrame,
    *,
    metric: str,
    final_database: str,
    from_col: str = "from_release",
    to_col: str = "to_release",
    final_db_col: str = "final_database",
) -> pd.DataFrame:
    """Return mean metric vs |Δrelease| split by direction (past vs future)."""

    sub = agg[agg[final_db_col] == final_database].copy()
    if sub.empty:
        return pd.DataFrame()

    sub[from_col] = sub[from_col].astype(int)
    sub[to_col] = sub[to_col].astype(int)
    sub["abs_delta"] = (sub[to_col] - sub[from_col]).abs()
    sub["direction"] = np.where(sub[to_col] >= sub[from_col], "future_or_same", "past")
    sub = sub[sub["abs_delta"] > 0]

    g = sub.groupby(["direction", "abs_delta"], as_index=False)[[metric]].mean(numeric_only=True)
    return g.sort_values(["direction", "abs_delta"]).reset_index(drop=True)


def release_stability_index(
    agg: pd.DataFrame,
    *,
    metric: str,
    final_database: str,
    axis: str = "from",
    from_col: str = "from_release",
    to_col: str = "to_release",
    final_db_col: str = "final_database",
) -> pd.DataFrame:
    """Summarize mean metric per release as a 'stability' index.

    axis:
      - 'from': for each source release r, mean over all target releases
      - 'to': for each target release r, mean over all source releases
    """

    sub = agg[agg[final_db_col] == final_database].copy()
    if sub.empty:
        return pd.DataFrame()

    if axis not in {"from", "to"}:
        raise ValueError("axis must be 'from' or 'to'")
    key = from_col if axis == "from" else to_col

    g = sub.groupby([key], as_index=False)[[metric]].mean(numeric_only=True).rename(columns={key: "release"})
    g["axis"] = axis
    return g.sort_values("release").reset_index(drop=True)

