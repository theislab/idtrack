"""Utilities for reproducibility/experiments notebooks and scripts.

This module is intentionally lightweight and avoids importing IDTrack itself.
It provides:
  - repo-relative path discovery (works from nested notebook folders)
  - standard locations for caches and exported artefacts
  - standard locations for experiment-local outputs
  - Matplotlib rcParams loading used across manuscript-ready figures
  - small plotting/style helpers (colors, palettes)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import time
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, TypeVar


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root by searching parents for expected markers.

    This supports two common layouts:
      - package-only: `<root>/idtrack` + `<root>/reproducibility`
      - umbrella:     `<root>/idtrack` + `<root>/idtrack/reproducibility`

    If `REPO_ROOT` is set and points to a valid root, it is preferred.
    If markers are not found, fall back to the current working directory.
    """

    env_root = os.environ.get("REPO_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "idtrack").is_dir() and (
            (candidate / "reproducibility").is_dir() or (candidate / "idtrack" / "reproducibility").is_dir()
        ):
            return candidate

    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "idtrack").is_dir() and (
            (candidate / "reproducibility").is_dir() or (candidate / "idtrack" / "reproducibility").is_dir()
        ):
            return candidate
    return here


def reproducibility_root(start: Path | None = None) -> Path:
    """Return the repo-relative `reproducibility/` root (layout-aware)."""

    root = find_repo_root(start)
    if (root / "reproducibility").is_dir():
        return root / "reproducibility"
    return root / "idtrack" / "reproducibility"


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def require_env_var(name: str, *, example: str | None = None) -> str:
    """Return a required environment variable (stripped) or raise a helpful error."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    value = os.environ.get(name, "").strip()
    if value:
        return value

    hint = f"Set it in your shell: `export {name}=...`"
    if example:
        hint = f"Set it in your shell, e.g. `export {name}={example}`"
    raise RuntimeError(f"Missing required environment variable {name!r}. {hint}")


def idtrack_cache_dir(start: Path | None = None, *, env_var: str = "IDTRACK_LOCAL_REPO") -> Path:
    """Resolve the IDTrack cache directory, preferring an env var when set."""

    raw = os.environ.get(env_var, "").strip()
    if raw:
        return ensure_dir(Path(raw).expanduser().resolve())
    root = find_repo_root(start)
    if (root / "idtrack" / "docs" / "_notebooks").is_dir():
        return ensure_dir(root / "idtrack" / "docs" / "_notebooks" / "idtrack_cache")
    return ensure_dir(root / "docs" / "_notebooks" / "idtrack_cache")


def experiments_outputs_dir(start: Path | None = None, *, folder_name: str = "_outputs") -> Path:
    """Directory for experiment-local outputs (kept out of the global IDTrack cache)."""

    repro_root = reproducibility_root(start)
    return ensure_dir(repro_root / "experiments" / folder_name)


def experiment_outputs_dir(
    experiment: str, start: Path | None = None, *, folder_name: str = "_outputs", safe: bool = True
) -> Path:
    """Return an experiment-specific output dir under `experiments/_outputs/<experiment>`."""

    base = experiments_outputs_dir(start, folder_name=folder_name)
    sub = safe_tag(experiment) if safe else str(experiment)
    return ensure_dir(base / sub)


def manuscript_figures_dir(start: Path | None = None) -> Path:
    """Directory for publication-ready figures (export location).

    Stored under `reproducibility/experiments/_outputs/_publication/figures/`.
    """

    return ensure_dir(experiments_outputs_dir(start) / "_publication" / "figures")


def manuscript_tables_dir(start: Path | None = None) -> Path:
    """Directory for publication-ready tables (export location).

    Stored under `reproducibility/experiments/_outputs/_publication/tables/`.
    """

    return ensure_dir(experiments_outputs_dir(start) / "_publication" / "tables")


def default_rcparams_pickle(start: Path | None = None) -> Path:
    """Return the path to the shared rcParams pickle used by experiments."""

    repro_root = reproducibility_root(start)
    return repro_root / "experiments" / "figure_rcparams" / "rcparams.pickle"


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


def experiment_figures_dir(ctx: "NotebookContext") -> Path:
    """Return an experiment-local figure dir under `experiments/_outputs/<experiment>/figures`."""

    return ensure_dir(ctx.experiment_outputs / "figures")


def experiment_tables_dir(ctx: "NotebookContext") -> Path:
    """Return an experiment-local table dir under `experiments/_outputs/<experiment>/tables`."""

    return ensure_dir(ctx.experiment_outputs / "tables")


def save_figure(
    fig,
    filename: str,
    ctx: "NotebookContext",
    *,
    dpi: int = 300,
    formats: tuple[str, ...] = ("pdf",),
    also_experiment_outputs: bool = True,
) -> dict[str, Path]:
    """Save a figure into the publication export dir (and optionally experiment-local outputs).

    Returns a dict of format -> written path (export location).
    """

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")

    name = filename.strip()
    if "." in name:
        stem = name.rsplit(".", 1)[0]
    else:
        stem = name

    written: dict[str, Path] = {}
    for fmt in formats:
        fmt = str(fmt).lstrip(".").lower()
        out = ctx.manuscript_figures / f"{stem}.{fmt}"
        written[fmt] = savefig(fig, out, dpi=dpi)

        if also_experiment_outputs:
            out2 = experiment_figures_dir(ctx) / f"{stem}.{fmt}"
            savefig(fig, out2, dpi=dpi)

    return written


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
    use_lock: bool = True,
    lock_timeout_s: float = 12 * 60 * 60,
    lock_stale_s: float = 24 * 60 * 60,
) -> _T:
    """Load a cached artifact if present, otherwise compute and cache it.

    Notes:
        - Uses an optional filesystem lock to make parallel Slurm runs safe
          when multiple notebooks may attempt to build the same cache file.
        - The lock is best-effort and designed to prevent duplicate work, not
          to provide strong transactional guarantees.
    """

    p = Path(path).expanduser().resolve()
    if p.exists() and not force:
        if load is None:
            raise ValueError("cache_or_compute: 'load' must be provided when reading from cache.")
        return load(p)

    lock_path = p.with_suffix(p.suffix + ".lock")
    ctx = file_lock(lock_path, timeout_s=lock_timeout_s, stale_s=lock_stale_s) if use_lock else nullcontext()
    with ctx:
        # Another worker might have finished while we waited.
        if p.exists() and not force:
            if load is None:
                raise ValueError("cache_or_compute: 'load' must be provided when reading from cache.")
            return load(p)

        obj = compute()
        if save is None:
            raise ValueError("cache_or_compute: 'save' must be provided when writing to cache.")
        save(obj, p)
        return obj


@contextmanager
def nullcontext():
    yield


@contextmanager
def file_lock(
    path: Path,
    *,
    timeout_s: float = 12 * 60 * 60,
    poll_s: float = 0.5,
    stale_s: float = 24 * 60 * 60,
) -> Any:
    """A minimal cross-platform filesystem lock using an exclusive lock file.

    Designed for Slurm-style parallel notebook execution where multiple workers may
    attempt to build the same cache artifact simultaneously.

    The lock is *advisory* and best-effort:
      - It prevents most duplicate work.
      - It is safe with atomic writes (see `atomic_write_bytes`).
      - It cannot protect against all failure modes (e.g. stale locks after crashes).

    Args:
        path: Path to the lock file (e.g. `<cache>.lock`).
        timeout_s: Max time to wait for the lock before raising TimeoutError.
        poll_s: Sleep duration between lock acquisition attempts.
        stale_s: Treat locks older than this as stale and remove them.
    """

    lock = Path(path).expanduser().resolve()
    ensure_dir(lock.parent)

    start = time.time()
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
            finally:
                os.close(fd)
                fd = None
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
                if stale_s and age > stale_s:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue

            if timeout_s and (time.time() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock}")
            time.sleep(poll_s)

    try:
        yield lock
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:  # noqa: S110
            pass


def cache_pickle_or_compute(
    path: Path,
    compute: Callable[[], _T],
    *,
    force: bool = False,
    use_lock: bool = True,
) -> _T:
    """cache_or_compute with pickle I/O helpers."""

    return cache_or_compute(path, compute, load=read_pickle, save=write_pickle, force=force, use_lock=use_lock)


def cache_json_or_compute(
    path: Path,
    compute: Callable[[], _T],
    *,
    force: bool = False,
    use_lock: bool = True,
) -> _T:
    """cache_or_compute with JSON I/O helpers."""

    return cache_or_compute(path, compute, load=read_json, save=write_json, force=force, use_lock=use_lock)


# ---------------------------------------------------------------------------
# Lightweight export helpers (DataFrames / tables)
# ---------------------------------------------------------------------------


def atomic_write_dataframe_csv(df, path: Path, *, index: bool = False, **to_csv_kwargs) -> Path:
    """Write a DataFrame to CSV with atomic semantics (via temp file + rename)."""

    # Import lazily to keep this module lightweight for non-notebook usage.
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, got {type(df)}")

    path = Path(path).expanduser().resolve()
    ensure_dir(path.parent)

    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_csv(tmp_path, index=index, **to_csv_kwargs)
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # noqa: S110
            pass

    return path


def atomic_write_dataframe_tsv(df, path: Path, *, index: bool = False, **to_csv_kwargs) -> Path:
    """Write a DataFrame to TSV with atomic semantics."""

    return atomic_write_dataframe_csv(df, path, index=index, sep="\t", **to_csv_kwargs)


def atomic_write_dataframe_latex(df, path: Path, *, index: bool = False, **to_latex_kwargs) -> Path:
    """Write a DataFrame to LaTeX with atomic semantics."""

    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, got {type(df)}")

    text = df.to_latex(index=index, **to_latex_kwargs)
    return atomic_write_text(Path(path), text + ("\n" if not text.endswith("\n") else ""))


# ---------------------------------------------------------------------------
# Manuscript plotting helpers (shared palette)
# ---------------------------------------------------------------------------

MANUSCRIPT_COLORS: dict[str, str] = {
    # Outcome colors (used consistently across experiments)
    "1→0": "#C44E52",
    "1→1": "#4C72B0",
    "1→n": "#DD8452",
    # Generic TDM/ATM breakdown (TDM = true match in final DB; ATM = Ensembl fallback due to missing synonym)
    "1→1 TDM": "#4C72B0",
    "1→1 ATM": "#9BB6D6",
    "1→n TDM": "#DD8452",
    "1→n ATM": "#F0B694",
    # HLCA detailed breakdown (HGNC target) — TDM vs ATM = fallback
    "HGNC 1→1 TDM": "#4C72B0",
    "HGNC 1→1 ATM": "#9BB6D6",
    "HGNC 1→n TDM": "#DD8452",
    "HGNC 1→n ATM": "#F0B694",
    # Neutral greys
    "neutral": "#999999",
    "grid": "#DDDDDD",
    # External tool palette (used in tool-comparison notebooks)
    "IDTrack": "#4C72B0",
    "pybiomart": "#55A868",
    "mygene": "#C44E52",
    "gprofiler": "#8172B3",
    "gget": "#CCB974",
}

EXTERNAL_MAPPER_METHODS_ORDERED: tuple[str, ...] = ("IDTrack", "pybiomart", "mygene", "gprofiler", "gget")


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


def external_method_palette(methods: list[str] | tuple[str, ...]) -> list[str]:
    """Return a stable palette for external-mapper method names."""

    return manuscript_palette(list(methods))


# ---------------------------------------------------------------------------
# Notebook bootstrap helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NotebookContext:
    """Resolved paths + shared style for experiment notebooks."""

    repo_root: Path
    experiments_src: Path
    idtrack_local_repo: Path
    experiment_cache: Path
    experiments_outputs: Path
    experiment_outputs: Path
    manuscript_figures: Path
    manuscript_tables: Path
    experiment: str


def apply_manuscript_style(*, seaborn: bool = True, rcparams_path: Path | None = None) -> None:
    """Apply shared Matplotlib (and optional seaborn) styling for manuscript-ready figures."""

    try:
        apply_rcparams(load_rcparams(rcparams_path))
    except Exception as exc:  # noqa: S110
        print(f"Warning: could not apply shared rcParams: {exc}")

    if seaborn:
        try:
            import seaborn as sns

            sns.set_theme(style="whitegrid", context="paper")
        except Exception:  # noqa: S110
            pass

    # Reasonable interactive defaults; export DPI is handled in `savefig`.
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.dpi": 140})
    except Exception:  # noqa: S110
        pass


def notebook_context(
    experiment: str,
    *,
    start: Path | None = None,
    apply_style: bool = True,
    seaborn: bool = True,
) -> NotebookContext:
    """Return commonly used paths for a notebook (cache dirs + export dirs)."""

    root = find_repo_root(start)
    experiments_src = reproducibility_root(root) / "experiments" / "src"

    ctx = NotebookContext(
        repo_root=root,
        experiments_src=experiments_src,
        idtrack_local_repo=idtrack_cache_dir(root),
        experiment_cache=experiments_cache_dir(root, experiment=experiment),
        experiments_outputs=experiments_outputs_dir(root),
        experiment_outputs=experiment_outputs_dir(experiment, root),
        manuscript_figures=manuscript_figures_dir(root),
        manuscript_tables=manuscript_tables_dir(root),
        experiment=str(experiment),
    )

    # Ensure downstream code (and user notebooks) see a consistent cache root by default.
    os.environ.setdefault("IDTRACK_LOCAL_REPO", str(ctx.idtrack_local_repo))

    if apply_style:
        apply_manuscript_style(seaborn=seaborn)

    return ctx


def safe_tag(value: str) -> str:
    """Return a filesystem-safe token used in cache filenames."""

    s = str(value)
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in s)


def stable_hash(text: str, *, n: int = 12) -> str:
    """Stable short hash for cache keys (sha256; not for security)."""

    if n <= 0:
        raise ValueError("n must be a positive integer")
    h = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return h[:n]


def fingerprint_list(values: list[str], *, n: int = 12, sep: str = "|") -> str:
    """Stable short hash for a list of values (order-sensitive)."""

    return stable_hash(sep.join(map(str, values)), n=n)


def write_csv_text(text: str, path: Path) -> Path:
    """Write CSV text with atomic semantics."""

    return atomic_write_text(path, text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure helpers (panel labels, consistent exports)
# ---------------------------------------------------------------------------


def label_panels(
    axes,
    *,
    labels: tuple[str, ...] | None = None,
    x: float = -0.06,
    y: float = 1.03,
    fontsize: int = 11,
    fontweight: str = "bold",
) -> None:
    """Label subplot panels as (a), (b), ... with a consistent style.

    Args:
        axes: A Matplotlib Axes, or an iterable of Axes.
        labels: Explicit labels to use; defaults to lowercase letters.
    """

    if axes is None:
        return

    if hasattr(axes, "__iter__") and not isinstance(axes, (str, bytes)):
        ax_list = list(axes)
    else:
        ax_list = [axes]

    if labels is None:
        alpha = "abcdefghijklmnopqrstuvwxyz"
        labels = tuple(f"({alpha[i]})" for i in range(min(len(ax_list), len(alpha))))

    for i, ax in enumerate(ax_list):
        if i >= len(labels):
            break
        try:
            ax.text(x, y, labels[i], transform=ax.transAxes, fontsize=fontsize, fontweight=fontweight, va="bottom")
        except Exception:  # noqa: S110
            pass
