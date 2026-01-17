"""Set-based agreement utilities for experiment notebooks.

Used for marketing-grade figures/tables where we want to compare:
  - dataset harmonization overlap (HLCA)
  - method agreement (external mappers vs IDTrack)
  - stability diagnostics (round-trip / time travel)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "pairwise_jaccard_matrix",
    "pairwise_overlap_counts",
    "set_size_summary",
]


def _jaccard(a: set[str], b: set[str]) -> float:
    denom = len(a | b)
    return (len(a & b) / denom) if denom else 1.0


def pairwise_jaccard_matrix(sets: Mapping[str, set[str]]) -> pd.DataFrame:
    """Return an NxN Jaccard similarity matrix for a mapping of name->set."""

    keys = list(sets.keys())
    n = len(keys)
    mat = np.zeros((n, n), dtype=float)
    for i, ki in enumerate(keys):
        ai = sets.get(ki, set())
        for j, kj in enumerate(keys):
            if i == j:
                mat[i, j] = 1.0
                continue
            bj = sets.get(kj, set())
            mat[i, j] = _jaccard(ai, bj)
    return pd.DataFrame(mat, index=keys, columns=keys)


def pairwise_overlap_counts(sets: Mapping[str, set[str]]) -> pd.DataFrame:
    """Return an NxN overlap count matrix (|Ai ∩ Aj|)."""

    keys = list(sets.keys())
    n = len(keys)
    mat = np.zeros((n, n), dtype=int)
    for i, ki in enumerate(keys):
        ai = sets.get(ki, set())
        for j, kj in enumerate(keys):
            bj = sets.get(kj, set())
            mat[i, j] = int(len(ai & bj))
    return pd.DataFrame(mat, index=keys, columns=keys)


def set_size_summary(sets: Mapping[str, set[str]]) -> pd.DataFrame:
    """Return a tidy per-key summary (size + emptiness)."""

    rows = []
    for k, s in sets.items():
        ss = s or set()
        rows.append({"name": str(k), "n": int(len(ss)), "empty": bool(len(ss) == 0)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["n", "name"], ascending=[False, True]).reset_index(drop=True)

