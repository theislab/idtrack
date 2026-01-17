"""Utilities for reproducibility/experiments notebooks and scripts.

This module is intentionally lightweight and avoids importing IDTrack itself.
It provides:
  - repo-relative path discovery (works from nested notebook folders)
  - standard locations for caches and manuscript outputs
  - Matplotlib rcParams loading used across manuscript-ready figures
  - small plotting/style helpers (colors, palettes)
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by searching parents for expected markers.

    Markers:
      - `idtrack/` (python package root)
      - `idtrack-manuscript/` (LaTeX manuscript root)

    If markers are not found, fall back to the current working directory.
    """

    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "idtrack").is_dir() and (candidate / "idtrack-manuscript").is_dir():
            return candidate
    return here


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def idtrack_cache_dir(start: Path | None = None, *, env_var: str = "IDTRACK_LOCAL_REPO") -> Path:
    """Resolve the IDTrack cache directory, preferring an env var when set."""

    raw = os.environ.get(env_var, "").strip()
    if raw:
        return ensure_dir(Path(raw).expanduser().resolve())
    root = find_repo_root(start)
    return ensure_dir(root / "idtrack" / "docs" / "_notebooks" / "idtrack_cache")


def experiments_outputs_dir(start: Path | None = None, *, folder_name: str = "_outputs") -> Path:
    """Directory for experiment-local outputs (kept out of the global IDTrack cache)."""

    root = find_repo_root(start)
    return ensure_dir(root / "idtrack" / "reproducibility" / "experiments" / folder_name)


def manuscript_figures_dir(start: Path | None = None) -> Path:
    """Return `idtrack-manuscript/figures`."""

    root = find_repo_root(start)
    return ensure_dir(root / "idtrack-manuscript" / "figures")


def manuscript_tables_dir(start: Path | None = None) -> Path:
    """Return `idtrack-manuscript/tables`."""

    root = find_repo_root(start)
    return ensure_dir(root / "idtrack-manuscript" / "tables")


def default_rcparams_pickle(start: Path | None = None) -> Path:
    """Return the path to the shared rcParams pickle used by experiments."""

    root = find_repo_root(start)
    return root / "idtrack" / "reproducibility" / "experiments" / "figure_rcparams" / "rcparams.pickle"


def load_rcparams(path: Path | None = None) -> dict[str, Any]:
    """Load Matplotlib rcParams from a pickle file."""

    p = path or default_rcparams_pickle()
    with open(p, "rb") as handle:
        rcparams = pickle.load(handle)  # noqa: S301
    if not isinstance(rcparams, dict):
        raise TypeError(f"Expected rcparams pickle to contain a dict, got {type(rcparams)}")
    return rcparams


def apply_rcparams(rcparams: dict[str, Any]) -> None:
    """Apply a rcParams dict to Matplotlib (imported lazily)."""

    import matplotlib.pyplot as plt

    plt.rcParams.update(rcparams)


def savefig(fig, path: Path, *, dpi: int = 300) -> Path:
    """Save a Matplotlib figure with consistent defaults and return the written path."""

    path = Path(path).expanduser().resolve()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def experiments_cache_dir(start: Path | None = None, *, experiment: str | None = None) -> Path:
    """Return an experiment-specific cache dir under the shared IDTrack cache."""

    base = ensure_dir(idtrack_cache_dir(start) / "experiments")
    if not experiment:
        return base
    return ensure_dir(base / experiment)


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Atomically write bytes to disk (safe against partial writes)."""

    path = Path(path).expanduser().resolve()
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write text to disk (safe against partial writes)."""

    return atomic_write_bytes(path, text.encode(encoding))


def read_pickle(path: Path) -> Any:
    """Load a pickle from disk."""

    with open(Path(path), "rb") as handle:
        return pickle.load(handle)  # noqa: S301


def write_pickle(obj: Any, path: Path) -> Path:
    """Atomically write a pickle to disk."""

    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return atomic_write_bytes(Path(path), data)


def read_json(path: Path) -> Any:
    """Load JSON from disk."""

    return json.loads(Path(path).read_text())


def write_json(obj: Any, path: Path, *, indent: int = 2, sort_keys: bool = True) -> Path:
    """Atomically write JSON to disk."""

    text = json.dumps(obj, indent=indent, sort_keys=sort_keys)
    return atomic_write_text(Path(path), text + "\n")


_T = TypeVar("_T")


def cache_or_compute(
    path: Path,
    compute: Callable[[], _T],
    *,
    load: Callable[[Path], _T] | None = None,
    save: Callable[[_T, Path], Path] | None = None,
    force: bool = False,
) -> _T:
    """Load a cached artifact if present, otherwise compute and cache it."""

    p = Path(path).expanduser().resolve()
    if p.exists() and not force:
        if load is None:
            raise ValueError("cache_or_compute: 'load' must be provided when reading from cache.")
        return load(p)

    obj = compute()
    if save is None:
        raise ValueError("cache_or_compute: 'save' must be provided when writing to cache.")
    save(obj, p)
    return obj


# ---------------------------------------------------------------------------
# Manuscript plotting helpers (shared palette)
# ---------------------------------------------------------------------------

MANUSCRIPT_COLORS: dict[str, str] = {
    # Outcome colors (used consistently across experiments)
    "1→0": "#C44E52",
    "1→1": "#4C72B0",
    "1→n": "#DD8452",
    # HLCA detailed breakdown (HGNC target) — TDM vs ATM = fallback
    "HGNC 1→1 TDM": "#4C72B0",
    "HGNC 1→1 ATM": "#9BB6D6",
    "HGNC 1→n TDM": "#DD8452",
    "HGNC 1→n ATM": "#F0B694",
    # Neutral greys
    "neutral": "#999999",
    "grid": "#DDDDDD",
}


def manuscript_palette(keys: list[str] | tuple[str, ...]) -> list[str]:
    """Return a list of colors for a list of semantic keys.

    Falls back to Matplotlib's default cycle for unknown keys.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: S110
        plt = None

    cycle: list[str] = []
    if plt is not None:
        try:
            cycle = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
        except Exception:  # noqa: S110
            cycle = []

    out: list[str] = []
    for i, k in enumerate(keys):
        if k in MANUSCRIPT_COLORS:
            out.append(MANUSCRIPT_COLORS[k])
        elif cycle:
            out.append(cycle[i % len(cycle)])
        else:
            out.append("#333333")
    return out
