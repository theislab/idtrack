"""Shared plotting helpers for manuscript-facing experiment notebooks.

Goals:
  - Make figure code concise inside notebooks
  - Keep visual style consistent across experiments (colors, grids, labels)
  - Provide seaborn-first plotting with a Matplotlib fallback
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from experiments_utils import MANUSCRIPT_COLORS

__all__ = [
    "heatmap",
    "stacked_barh",
    "errorbar_lines",
]


def _try_import_seaborn():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:  # noqa: S110
        return None


def heatmap(
    ax,
    data: pd.DataFrame,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    square: bool = True,
    cbar: bool = True,
    cbar_label: str | None = None,
    center: float | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    linewidths: float = 0.25,
    linecolor: str | None = None,
) -> Any:
    """Draw a heatmap with seaborn fallback and consistent cosmetics."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"heatmap: expected pandas.DataFrame, got {type(data)}")

    sns = _try_import_seaborn()
    linecolor = linecolor or MANUSCRIPT_COLORS["grid"]

    if sns is not None:
        hm = sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            cbar=cbar,
            square=square,
            annot=annot,
            fmt=fmt,
            linewidths=linewidths,
            linecolor=linecolor,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        if cbar and cbar_label and hm.collections:
            cb = hm.collections[0].colorbar
            if cb is not None:
                cb.set_label(cbar_label)
    else:
        aspect = "equal" if square else "auto"
        im = ax.imshow(data.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels([str(c) for c in data.columns], rotation=45, ha="right")
        ax.set_yticklabels([str(i) for i in data.index], rotation=0)
        if cbar:
            cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cb.set_label(cbar_label)

    ax.set_title(title)
    return ax


def stacked_barh(
    ax,
    df: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    xlim: tuple[float, float] | None = (0.0, 1.0),
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Horizontal stacked bar plot for a per-row fraction table."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"stacked_barh: expected pandas.DataFrame, got {type(df)}")
    if df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    legend_kwargs = legend_kwargs or {}
    left = None
    for col in df.columns:
        c = None if colors is None else colors.get(col)
        ax.barh(df.index, df[col].values, left=left, label=str(col), color=c)
        left = df[col].values if left is None else (np.asarray(left) + df[col].values)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=True, **legend_kwargs)
    return ax


def errorbar_lines(
    ax,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    yerr: str | None = None,
    hue: str,
    order: Iterable[str] | None = None,
    palette: dict[str, str] | None = None,
    fmt: str = "-o",
    lw: float = 1.2,
    ms: float = 3.0,
) -> Any:
    """Plot multiple lines with optional error bars (grouped by `hue`)."""

    if df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    groups = list(order) if order is not None else sorted(df[hue].dropna().astype(str).unique().tolist())
    for g in groups:
        sub = df[df[hue].astype(str) == str(g)]
        if sub.empty:
            continue
        sub = sub.sort_values(x)
        color = palette.get(str(g)) if palette else None
        ax.errorbar(
            sub[x],
            sub[y],
            yerr=sub[yerr] if (yerr and yerr in sub.columns) else None,
            fmt=fmt,
            lw=lw,
            ms=ms,
            label=str(g),
            color=color,
        )
    return ax

