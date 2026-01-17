"""Helpers for working with cached IDTrack conversion outputs in experiment notebooks.

This module is intentionally graph-free: it operates on the dictionaries returned by
`idtrack.API.convert_identifier` / `convert_identifier_multiple` without requiring `build_graph()`.

Core use cases:
- Summarize conversions into 1→0 / 1→1 / 1→n buckets
- Separate "target database matching" (TDM) from "alternative target" (ATM) via `no_target`
- Convert matchings into a tidy pandas DataFrame for plotting and tables
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = [
    "classify_multiple_conversion",
    "summarize_binned_conversion",
    "matchings_to_frame",
]


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(value)
    except Exception:  # noqa: S110
        return False


def classify_multiple_conversion(matchings: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Pure-function replica of `idtrack.API.classify_multiple_conversion`.

    The returned bins intentionally overlap (e.g., a changed 1→1 mapping also appears in `matching_1_to_1`).
    """

    result: dict[str, list[dict[str, Any]]] = {
        "changed_only_1_to_n": [],
        "changed_only_1_to_1": [],
        "alternative_target_1_to_1": [],
        "alternative_target_1_to_n": [],
        "matching_1_to_0": [],
        "matching_1_to_1": [],
        "matching_1_to_n": [],
        "input_identifiers": [],
    }

    for item in matchings:
        result["input_identifiers"].append(item)

        no_corresponding = _as_bool(item.get("no_corresponding"))
        no_conversion = _as_bool(item.get("no_conversion"))
        no_target = _as_bool(item.get("no_target"))

        if no_corresponding or no_conversion:
            result["matching_1_to_0"].append(item)
            continue

        targets = item.get("target_id") or []
        if len(targets) == 0:
            raise ValueError(
                f"Unexpected conversion result: query_id={item.get('query_id')!r} returned an empty target_id list "
                "despite no_corresponding/no_conversion being False."
            )

        query_id = item.get("query_id")

        if no_target:
            if len(targets) == 1:
                result["alternative_target_1_to_1"].append(item)
            else:
                result["alternative_target_1_to_n"].append(item)
        else:
            if len(targets) == 1 and targets[0] != query_id:
                result["changed_only_1_to_1"].append(item)

            if len(targets) > 1 and not any(query_id == k for k in targets):
                result["changed_only_1_to_n"].append(item)

            if len(targets) == 1:
                result["matching_1_to_1"].append(item)

            if len(targets) > 1:
                result["matching_1_to_n"].append(item)

    return result


def summarize_binned_conversion(classified: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Return a compact count summary for a classified conversion payload."""

    def _n(key: str) -> int:
        return len(classified.get(key, []) or [])

    total = _n("input_identifiers")
    out = {
        "total": total,
        "1_to_0": _n("matching_1_to_0"),
        "1_to_1_tdm": _n("matching_1_to_1"),
        "1_to_1_atm": _n("alternative_target_1_to_1"),
        "1_to_n_tdm": _n("matching_1_to_n"),
        "1_to_n_atm": _n("alternative_target_1_to_n"),
        "changed_1_to_1": _n("changed_only_1_to_1"),
        "changed_1_to_n": _n("changed_only_1_to_n"),
    }
    out["1_to_1_total"] = out["1_to_1_tdm"] + out["1_to_1_atm"]
    out["1_to_n_total"] = out["1_to_n_tdm"] + out["1_to_n_atm"]
    return out


def matchings_to_frame(matchings: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of IDTrack matchings into a tidy DataFrame (one row per query)."""

    rows: list[dict[str, Any]] = []
    for item in matchings:
        no_corresponding = _as_bool(item.get("no_corresponding"))
        no_conversion = _as_bool(item.get("no_conversion"))
        no_target = _as_bool(item.get("no_target"))
        query_id = item.get("query_id")
        targets = item.get("target_id") or []

        if no_corresponding or no_conversion:
            mapping = "1:0"
            n_targets = 0
        else:
            if len(targets) == 0:
                raise ValueError(
                    f"Unexpected conversion result: query_id={query_id!r} returned empty target_id list despite "
                    "no_corresponding/no_conversion being False."
                )
            n_targets = len({str(t) for t in targets})
            mapping = "1:1" if n_targets == 1 else "1:n"

        changed_1_to_1 = False
        changed_1_to_n = False
        if mapping == "1:1" and not no_target and targets and targets[0] != query_id:
            changed_1_to_1 = True
        if mapping == "1:n" and not no_target and targets and not any(query_id == k for k in targets):
            changed_1_to_n = True

        rows.append(
            {
                "query_id": query_id,
                "final_database": item.get("final_database"),
                "mapping": mapping,
                "n_targets": n_targets,
                "no_corresponding": no_corresponding,
                "no_conversion": no_conversion,
                "no_target": no_target,
                "changed_only_1_to_1": changed_1_to_1,
                "changed_only_1_to_n": changed_1_to_n,
            }
        )

    return pd.DataFrame(rows)

