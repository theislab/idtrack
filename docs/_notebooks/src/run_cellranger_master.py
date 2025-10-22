
#!/usr/bin/env python3
"""
run_cellranger_master.py

Master function to run either `cellranger count` or `cellranger multi` for ONE dataset,
using the Ensembl reference built earlier by `build_cellranger_mkref` and the FASTQ
specification from `fastq_datasets.py` (and files downloaded via `download_fastq.py`).

Key guarantees
--------------
- Chooses exactly one of `cellranger count` or `cellranger multi` based on the dataset's
  `mode` field ("count" or "multi"). It will not run both.
- Uses the reference folder created by `build_cellranger_mkref` for the specified Ensembl
  release (defaults to assembly_label="GRCh38", path pattern:
  ``<workdir>/GRCh38_<release>/reference_GRCh38_<release>``). If a persisted job-status JSON
  exists (written by `build_cellranger_mkref`) it is consulted first.
- Places outputs under: ``<results_dir>/<dataset_key>/<TAG>/`` and runs Cell Ranger with
  cwd set to that directory to keep everything contained there.
  The tag defaults to: ``GHRc{ENSEMBL_RELEASE}_{CR_MAJOR}{CR_MINOR}`` (e.g., GHRc110_90).
- Ensures FASTQs are present; if needed, downloads them via `download_fastq.download_fastq(...)`,
  then extracts .tar/.tar.gz archives. It locates one or more FASTQ directories automatically.
- For `multi`, if a `multi_csv` aux file is present in the dataset spec, it is downloaded
  (if needed) and patched so that:
    * [gene expression] reference -> points to the mkref directory
    * [libraries] fastqs -> point to the discovered FASTQ directories (semicolon-separated)
    * [feature] feature_ref -> can be patched when provided in `aux` as "feature_ref"
- For `count`, it auto-fills `--transcriptome`, `--fastqs`, and tries to infer `--sample`
  (comma-separated) from FASTQ filenames if not provided through `params`.

Inputs
------
- ONE dataset key (string) present in `fastq_datasets.fastq_datasets`
- ONE Ensembl release number (int), e.g., 110
- A working directory (Path-like) used by the prior scripts
- A results directory (Path-like) where we create: results/<dataset_key>/<TAG>/

Outputs
-------
A structured dictionary describing the run, including:
  {
    "dataset_key": ...,
    "mode": "count" | "multi",
    "cellranger": "/abs/path/to/cellranger",
    "cellranger_version": "9.0.1",
    "release": 110,
    "assembly_label": "GRCh38",
    "reference_dir": "/abs/path/to/reference_GRCh38_110",
    "results_root": "/abs/path/to/<results>/<dataset>/<TAG>",
    "run_id": "<derived id>",
    "cmd": [...],
    "returncode": 0/!=0,
    "stdout_log": ".../cellranger_stdout.log",
    "stderr_log": ".../cellranger_stderr.log",
    "run_dir": ".../<TAG>/<run_id>",
    "outs_dir": ".../<TAG>/<run_id>/outs" (if exists),
  }

Usage example
-------------
>>> from run_cellranger_master import run_cellranger_master
>>> summary = run_cellranger_master(
...     dataset_key="pbmc_1k_v3",
...     release=110,
...     workdir="/data/work",
...     results_dir="/data/results",
...     # Optional: if "cellranger" is not on PATH, point to it explicitly:
...     # cellranger_bin="/opt/apps/cellranger-9.0.1/cellranger",
... )

Notes
-----
- This script does not attempt to *install* Cell Ranger. Use `install_cellranger.install_cellranger(...)`
  beforehand and either put the launcher on PATH or pass its absolute path as `cellranger_bin`.
- This script does not run `build_cellranger_mkref` itself. It *uses* its outputs by locating the
  reference folder. If missing or incomplete, it raises a clear error.
- This script *only* runs one of `count` or `multi` per call (never both).

"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

# Local modules created previously by you (imported but not executed here)
from fastq_datasets import fastq_datasets, DatasetSpec  # type: ignore
from download_fastq import download_fastq  # type: ignore


# --------------------------- Utilities ---------------------------

def _exists_nonempty(p: Optional[Path]) -> bool:
    try:
        return bool(p) and Path(p).exists() and Path(p).stat().st_size > 0
    except Exception:
        return False


def _resolve_cellranger(cellranger_bin: Path | str | None = None) -> Path:
    """
    Resolve path to the 'cellranger' launcher.

    Resolution order:
    1) Explicit `cellranger_bin` if provided (file or directory containing `cellranger`).
    2) Environment variable CELLRANGER_BIN (file or directory).
    3) Fallback to `cellranger` resolved from PATH.

    Ensures the final path exists and is executable.
    """
    cand: Optional[Path] = None

    def normalize(x: str | Path) -> Path:
        p = Path(x)
        return (p / "cellranger") if p.is_dir() else p

    if cellranger_bin:
        cand = normalize(cellranger_bin)
    elif os.environ.get("CELLRANGER_BIN"):
        cand = normalize(os.environ["CELLRANGER_BIN"])
    else:
        which = shutil.which("cellranger")
        if which:
            cand = Path(which)

    if not cand:
        raise FileNotFoundError(
            "Could not resolve 'cellranger'. Provide `cellranger_bin` or set CELLRANGER_BIN."
        )
    if not cand.exists():
        raise FileNotFoundError(f"cellranger not found at: {cand}")
    if not os.access(str(cand), os.X_OK):
        raise PermissionError(f"cellranger exists but is not executable: {cand}")
    return cand.resolve()


def _cellranger_version(cellranger: Path) -> str:
    """Return '9.0.1' from 'cellranger-9.0.1' output. Returns 'unknown' on failure."""
    try:
        cp = subprocess.run([str(cellranger), "--version"], capture_output=True, text=True, check=False)
        out = (cp.stdout or cp.stderr or "").strip()
        # Expected like: 'cellranger-9.0.1'
        m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", out)
        return m.group(0) if m else "unknown"
    except Exception:
        return "unknown"


def _mk_alignment_tag(release: int, cellranger_version: str, prefix: str = "GHRc") -> str:
    """
    Compose the alignment tag subfolder name, e.g.:
      GHRc110_90   -> Ensembl release 110, Cell Ranger 9.0.x
    """
    # Parse MAJOR.MINOR (fallback '00')
    m = re.match(r"^(\d+)\.(\d+)", cellranger_version)
    mm = (m.group(1) + m.group(2)) if m else "00"
    return f"{prefix}{release}_{mm}"


def _reference_dir(workdir: Path, release: int, assembly_label: str = "GRCh38") -> Path:
    """
    Locate the mkref output folder produced by build_cellranger_mkref:
      <workdir> / f"{assembly_label}_{release}" / f"reference_{assembly_label}_{release}"
    If a persisted job JSON exists, trust its 'mkref_dir' when present.
    """
    rel_dir = workdir / f"{assembly_label}_{release}"
    candidate = rel_dir / f"reference_{assembly_label}_{release}"
    job_file = candidate / "job_exit_status_message"
    if job_file.exists():
        try:
            payload = json.loads(job_file.read_text())
            s = payload.get("mkref_dir")
            if s:
                p = Path(s)
                if (p / "reference.json").exists():
                    return p.resolve()
        except Exception:
            pass  # fall back to deterministic path

    # Fallback: deterministic location
    if (candidate / "reference.json").exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find a complete Cell Ranger reference for release {release} at {candidate}."
        " Make sure `build_cellranger_mkref(...)` finished successfully."
    )


def _extract_if_needed(archive: Path, dest_dir: Path) -> None:
    """Extract .tar or .tar.gz archives into dest_dir (idempotent)."""
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    # If already extracted (we detect .fastq.gz present), skip
    fastqs_found = list(dest_dir.rglob("*.fastq.gz"))
    if fastqs_found:
        return
    suffixes = "".join(archive.suffixes)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if suffixes in (".tar", ".tar.gz", ".tgz", ".tar.gzip"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format for FASTQs: {archive.name} (suffixes={archive.suffixes})")


def _discover_fastq_dirs(root: Path) -> list[Path]:
    """Return unique parent directories that contain FASTQ files under root."""
    parents: list[Path] = []
    seen: set[Path] = set()
    for fq in root.rglob("*.fastq.gz"):
        p = fq.parent.resolve()
        if p not in seen:
            seen.add(p)
            parents.append(p)
    if not parents:
        raise FileNotFoundError(f"No .fastq.gz files found under: {root}")
    return parents


def _guess_samples_from_fastqs(fastq_dirs: Iterable[Path], limit: int = 8) -> list[str]:
    """
    Infer sample prefixes from FASTQ filenames (10x style).
    Returns up to `limit` unique prefixes. If none inferred, return [].

    Examples matched (prefix extracted between start and first '_S' or '_L00'):
      Sample123_S1_L001_R1_001.fastq.gz   -> Sample123
      pbmc_1k_v3_S1_L002_R2_001.fastq.gz  -> pbmc_1k_v3
    """
    rx = re.compile(r"^(.+?)_(?:S\d+|L\d{3})_")
    samples: list[str] = []
    seen: set[str] = set()
    for d in fastq_dirs:
        for fq in d.glob("*.fastq.gz"):
            m = rx.match(fq.name)
            if m:
                s = m.group(1)
                if s not in seen:
                    seen.add(s)
                    samples.append(s)
                    if len(samples) >= limit:
                        return samples
    return samples


def _patch_multi_csv(src_csv: Path, dst_csv: Path, *, reference_dir: Path, fastq_dirs: list[Path], feature_ref: Optional[Path] = None) -> None:
    """
    Patch a 10x `cellranger multi` config CSV to inject reference and FASTQ locations.
    - For the [gene expression] section, set 'reference' to `reference_dir`.
    - For the [feature] section, if `feature_ref` is provided, set 'feature_ref'.
    - For the [libraries] rows, update 'fastqs' to the semicolon-separated list of discovered fastq_dirs.

    The format consists of bracketed sections and simple key,value lines, plus a [libraries] table.
    Non-recognized lines are preserved verbatim.
    """
    lines = src_csv.read_text().splitlines()
    out: list[str] = []
    section = None
    fq_str = ";".join(str(p) for p in fastq_dirs)

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            out.append(raw)
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            out.append(raw)
            continue

        # Key,Value pairs
        if "," in line and section in {"gene expression", "feature"}:
            key, val = [t.strip() for t in line.split(",", 1)]
            lk = key.lower()
            if section == "gene expression" and lk == "reference":
                out.append(f"reference,{reference_dir}")
                continue
            if section == "feature" and feature_ref is not None and lk == "feature_ref":
                out.append(f"feature_ref,{feature_ref}")
                continue
            # default: keep as-is
            out.append(raw)
            continue

        # [libraries] table: first line after header is the CSV header row
        if section == "libraries":
            # Replace the fastqs column in data rows (not the header)
            # We detect header by presence of the literal "fastqs" token.
            if "fastqs" in line.split(","):
                out.append(raw)  # header
            else:
                # Format: fastq_id,fastqs,sample,library_type,...
                parts = [t.strip() for t in line.split(",")]
                if len(parts) >= 2:
                    parts[1] = fq_str
                    out.append(",".join(parts))
                else:
                    out.append(raw)
            continue

        # default passthrough
        out.append(raw)

    dst_csv.write_text("\n".join(out))


def _coerce_cli_params(params: Mapping[str, Any]) -> list[str]:
    """
    Convert a mapping of Cell Ranger CLI-like parameters into ['--key=value', ...].
    Rules:
      - keys should already be in kebab-case ('expect-cells', 'chemistry', etc.).
      - bool True -> '--key' (flag), bool False -> omit
      - everything else -> '--key=value'
    """
    cli: list[str] = []
    for k, v in params.items():
        key = str(k).strip()
        if not key:
            continue
        if isinstance(v, bool):
            if v:
                cli.append(f"--{key}")
        else:
            cli.append(f"--{key}={v}")
    return cli


@dataclass
class RunSummary:
    dataset_key: str
    mode: str
    cellranger: Path
    cellranger_version: str
    release: int
    assembly_label: str
    reference_dir: Path
    results_root: Path
    run_id: str
    cmd: list[str]
    returncode: int
    stdout_log: Path
    stderr_log: Path
    run_dir: Optional[Path] = None
    outs_dir: Optional[Path] = None


# --------------------------- Main orchestrator ---------------------------

def run_cellranger_master(
    *,
    dataset_key: str,
    release: int,
    workdir: str | Path,
    results_dir: str | Path,
    cellranger_bin: str | Path | None = None,
    assembly_label: str = "GRCh38",
    download_if_missing: bool = True,
    resume_downloads: bool = True,
    use_threads: Optional[int] = None,  # reserved for future (could set --localcores)
    silent: bool = False,
) -> dict[str, Any]:
    """
    Orchestrate a single Cell Ranger run for `dataset_key` with Ensembl `release`.

    Parameters
    ----------
    dataset_key : str
        Key in `fastq_datasets.fastq_datasets` identifying the dataset spec.
    release : int
        Ensembl release number (e.g., 110) corresponding to the reference built earlier.
    workdir : Path-like
        Working directory used by prior scripts (`download_references.py`, `build_cellranger_mkref.py`,
        and `download_fastq.py`). FASTQs and refs are expected under here.
    results_dir : Path-like
        Where to create <results_dir>/<dataset_key>/<TAG>/ and place all outputs.
    cellranger_bin : Optional[str|Path]
        If omitted, we will try $CELLRANGER_BIN or look for 'cellranger' on PATH.
    assembly_label : str
        Assembly label used when building references (default "GRCh38").
    download_if_missing : bool
        If True, attempt to download FASTQs and aux files when missing using `download_fastq`.
    resume_downloads : bool
        If True, pass resume=True to the downloader.
    use_threads : Optional[int]
        Reserved for future expansions (e.g., --localcores); currently unused here.
    silent : bool
        Reduce console output if True.

    Returns
    -------
    dict[str, Any]
        JSON-serializable summary of the run; see `RunSummary` fields.
    """
    workdir = Path(workdir).expanduser().resolve()
    results_dir = Path(results_dir).expanduser().resolve()

    if dataset_key not in fastq_datasets:
        raise KeyError(f"Dataset key not found in fastq_datasets: {dataset_key!r}")
    ds_spec: Mapping[str, Any] = fastq_datasets[dataset_key]

    # Resolve cellranger
    cellranger = _resolve_cellranger(cellranger_bin)
    cr_ver = _cellranger_version(cellranger)

    # Locate reference directory
    ref_dir = _reference_dir(workdir, release, assembly_label=assembly_label)

    # Ensure FASTQs present (download if requested)
    ds_dir = workdir / dataset_key
    ds_dir.mkdir(parents=True, exist_ok=True)

    fastq_archive = None
    multi_csv_path = None
    feature_ref_path = None

    # If download desired or files not present, call downloader
    if download_if_missing or not any(ds_dir.iterdir()):
        try:
            overall = download_fastq({dataset_key: ds_spec}, working_dir=workdir, resume=resume_downloads, silent=silent)
            # Expected structure: overall[dataset_key]["files"]["fastqs"]["file"]
            ds_res = overall.get(dataset_key, {})
            files = ds_res.get("files", {})
            fq_info = files.get("fastqs", {})
            fastq_archive = Path(fq_info.get("file", "")) if fq_info else None

            aux = ds_spec.get("aux", {})
            if isinstance(aux, Mapping):
                # multi_csv and optional feature_ref
                if "multi_csv" in files:
                    multi_csv_path = Path(files["multi_csv"].get("file", "")) or None
                elif isinstance(aux.get("multi_csv"), Mapping):
                    # If downloader skipped, build expected path
                    multi_csv_path = ds_dir / str(aux["multi_csv"].get("filename", ""))

                if "feature_ref" in files:
                    feature_ref_path = Path(files["feature_ref"].get("file", "")) or None
                elif isinstance(aux.get("feature_ref"), Mapping):
                    feature_ref_path = ds_dir / str(aux["feature_ref"].get("filename", ""))

        except Exception as e:
            if not silent:
                print(f"[warn] FASTQ download step raised: {e}. Continuing to look for local files.")

    # If archive was not discovered via downloader, try to find it directly
    if fastq_archive is None:
        # pick the first known archive in ds_dir
        candidates = list(ds_dir.glob("*.tar")) + list(ds_dir.glob("*.tar.gz")) + list(ds_dir.glob("*.tgz"))
        fastq_archive = candidates[0] if candidates else None

    # If an archive exists, extract (idempotent). If there are already FASTQs, this is fast.
    if fastq_archive is not None and fastq_archive.exists():
        _extract_if_needed(fastq_archive, ds_dir)

    # Identify FASTQ directories (one or more)
    fastq_dirs = _discover_fastq_dirs(ds_dir)

    # Determine mode (count or multi)
    mode = str(ds_spec.get("mode", "count")).strip().lower()
    if mode not in {"count", "multi"}:
        raise ValueError(f"Unsupported dataset 'mode': {mode!r}. Expected 'count' or 'multi'.")

    # Build results root and run ID
    tag = _mk_alignment_tag(release, cr_ver, prefix="GHRc")
    results_root = (results_dir / dataset_key / tag).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    # Construct command
    run_id = f"{dataset_key}_{mode}_r{release}"
    params_map: Mapping[str, Any] = ds_spec.get("params", {}) if isinstance(ds_spec.get("params"), Mapping) else {}

    stdout_log = results_root / "cellranger_stdout.log"
    stderr_log = results_root / "cellranger_stderr.log"
    cmd: list[str] = []

    # Common helper: emit CLI params
    extra_cli = _coerce_cli_params(params_map)

    if mode == "count":
        # Auto-fill transcriptome, fastqs, and (optionally) sample(s)
        if not any(k.startswith("transcriptome") for k in params_map.keys()):
            extra_cli = [f"--transcriptome={ref_dir}"] + extra_cli

        # FASTQ directories (comma-separated)
        fastq_arg = ",".join(str(p) for p in fastq_dirs)
        extra_cli = [f"--fastqs={fastq_arg}"] + extra_cli

        # If --sample not provided, infer from filenames
        if not any(k == "sample" for k in params_map.keys()):
            inferred = _guess_samples_from_fastqs(fastq_dirs)
            if inferred:
                extra_cli = [f"--sample={','.join(inferred)}"] + extra_cli

        # If expected_cells not provided via params, use spec value (if any)
        if "expect-cells" not in params_map and ds_spec.get("expected_cells"):
            extra_cli = [f"--expect-cells={ds_spec['expected_cells']}"] + extra_cli

        # Chemistry hint from spec if available (and not overridden)
        if "chemistry" not in params_map and ds_spec.get("chemistry"):
            extra_cli = [f"--chemistry={ds_spec['chemistry']}"] + extra_cli

        cmd = [str(cellranger), "count", f"--id={run_id}"] + extra_cli

    else:  # mode == "multi"
        # Need a multi CSV. Prefer a downloaded aux file, else find a CSV in ds_dir.
        if multi_csv_path is None:
            # Fall back: any .csv that looks like a multi config
            csv_candidates = [p for p in ds_dir.glob("*.csv")]
            if not csv_candidates:
                raise FileNotFoundError(
                    "No multi CSV found. Provide it in the dataset 'aux' as 'multi_csv' or place a config.csv in the dataset folder."
                )
            multi_csv_path = csv_candidates[0]

        # Patch CSV to point to our reference and fastqs
        patched_csv = results_root / f"{Path(multi_csv_path).stem}.patched.csv"
        _patch_multi_csv(
            multi_csv_path,
            patched_csv,
            reference_dir=ref_dir,
            fastq_dirs=fastq_dirs,
            feature_ref=feature_ref_path,
        )

        cmd = [str(cellranger), "multi", f"--id={run_id}", f"--csv={patched_csv}"] + extra_cli

    # Persist the assembled command for reproducibility
    (results_root / "cellranger_cmd.txt").write_text(" ".join(cmd))

    # Run cellranger (in results_root). This will create a subfolder named after --id there.
    if not silent:
        print(f"[info] Running: {' '.join(cmd)}")
        print(f"[info]   cwd:   {results_root}")
    proc = subprocess.run(cmd, cwd=str(results_root), text=True, capture_output=True)

    stdout_log.write_text(proc.stdout or "")
    stderr_log.write_text(proc.stderr or "")

    # Derive run_dir and outs_dir
    run_dir = results_root / run_id
    outs_dir = run_dir / "outs"
    summary = RunSummary(
        dataset_key=dataset_key,
        mode=mode,
        cellranger=cellranger,
        cellranger_version=cr_ver,
        release=release,
        assembly_label=assembly_label,
        reference_dir=ref_dir,
        results_root=results_root,
        run_id=run_id,
        cmd=cmd,
        returncode=proc.returncode,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        run_dir=run_dir if run_dir.exists() else None,
        outs_dir=outs_dir if outs_dir.exists() else None,
    )

    # Return as plain dict (for JSON serialization)
    out = {
        "dataset_key": summary.dataset_key,
        "mode": summary.mode,
        "cellranger": str(summary.cellranger),
        "cellranger_version": summary.cellranger_version,
        "release": summary.release,
        "assembly_label": summary.assembly_label,
        "reference_dir": str(summary.reference_dir),
        "results_root": str(summary.results_root),
        "run_id": summary.run_id,
        "cmd": summary.cmd,
        "returncode": summary.returncode,
        "stdout_log": str(summary.stdout_log),
        "stderr_log": str(summary.stderr_log),
        "run_dir": str(summary.run_dir) if summary.run_dir else None,
        "outs_dir": str(summary.outs_dir) if summary.outs_dir else None,
    }
    return out


# Optional CLI
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Run cellranger (count or multi) for one dataset.")
    ap.add_argument("--dataset-key", required=True, help="Key in fastq_datasets.fastq_datasets")
    ap.add_argument("--release", required=True, type=int, help="Ensembl release (e.g., 110)")
    ap.add_argument("--workdir", required=True, help="Working directory used by prior scripts")
    ap.add_argument("--results-dir", required=True, help="Directory to store outputs under <dataset>/<TAG>/")
    ap.add_argument("--cellranger-bin", default=None, help="Path to cellranger launcher or folder (optional)")
    ap.add_argument("--assembly-label", default="GRCh38", help="Assembly label used by mkref (default GRCh38)")
    ap.add_argument("--no-download", action="store_true", help="Do not attempt downloads; require local files")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = ap.parse_args()

    try:
        summary = run_cellranger_master(
            dataset_key=args.dataset_key,
            release=args.release,
            workdir=args.workdir,
            results_dir=args.results_dir,
            cellranger_bin=args.cellranger_bin,
            assembly_label=args.assembly_label,
            download_if_missing=not args.no_download,
            silent=args.quiet,
        )
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary.get("returncode", 1) == 0 else 1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
