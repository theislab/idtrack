"""Analysis helpers for external mapper experiments.

Consumes summary tables produced by notebooks (cache build + aggregation) and provides:
  - scenario deltas (e.g. naive vs time-travel-assisted)
  - small, reusable transformations for plotting and tables
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "scenario_delta_table",
]


def scenario_delta_table(
    agg: pd.DataFrame,
    *,
    scenario_a: str,
    scenario_b: str,
    keys: list[str],
    metrics: list[str],
    scenario_col: str = "scenario",
    suffix_a: str = "_a",
    suffix_b: str = "_b",
) -> pd.DataFrame:
    """Compute per-key deltas for a pair of scenarios.

    Returns a wide table with columns:
      - <metric><suffix_a>, <metric><suffix_b>, delta_<metric> (= b - a)
    """

    if agg.empty:
        return pd.DataFrame()

    base_cols = list(dict.fromkeys([scenario_col, *keys, *metrics]))
    a = agg[agg[scenario_col] == scenario_a][base_cols].copy()
    b = agg[agg[scenario_col] == scenario_b][base_cols].copy()

    if a.empty or b.empty:
        return pd.DataFrame()

    a = a.drop(columns=[scenario_col]).rename(columns={m: f"{m}{suffix_a}" for m in metrics})
    b = b.drop(columns=[scenario_col]).rename(columns={m: f"{m}{suffix_b}" for m in metrics})

    out = a.merge(b, on=keys, how="inner")
    for m in metrics:
        out[f"delta_{m}"] = out[f"{m}{suffix_b}"] - out[f"{m}{suffix_a}"]
    return out

