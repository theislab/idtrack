#!/usr/bin/env python3
"""
build_cellranger_mkref.py

Create Cell Ranger references for Ensembl GRCh38 releases using files prepared by
download_references.download_references(...).

Behavior
--------
- For each release:
  * Require UNZIPPED GTF and FASTA paths (from download_references).
  * Create a filtered GTF that keeps ONLY protein_coding entries, placed next to the source GTF.
    Name pattern: <original>.filtered.gtf (e.g., Homo_sapiens.GRCh38.110.filtered.gtf)
  * Run `cellranger mkref` in the corresponding release folder (e.g., GRCh38_110) with:
      --genome=cellranger_mkref_GRCh38_<release>
      --fasta=<unzipped fasta>
      --genes=<filtered gtf>
      --nthreads=<80% of CPUs by default>
  * Skip steps that are already done and report what was skipped.

- NEW: If the mkref output directory already contains a 'job_exit_status_message'
  (created after any previous success or failure), the release is *restored from that
  file* and the work is skipped. This guarantees that running the script a second time
  returns an identical results dictionary.

- At the end, prints how many releases finished successfully in this run (status == "done").
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Optional, TypedDict

# --------------------------- Utilities ---------------------------


def _log(silent: bool, msg: str) -> None:
    if not silent:
        print(msg)


def _exists_nonempty(p: Path | None) -> bool:
    if not p:
        return False
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """
    Run a command and capture output (stdout/stderr). Returns CompletedProcess.
    """
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)


def _resolve_cellranger(cellranger_bin: Path | str) -> Path:
    """
    Resolve the path to the 'cellranger' executable.
    - If a directory is provided, append 'cellranger'.
    - If a name is provided, try PATH with shutil.which.
    - Ensure the file exists and is executable.
    """
    p = Path(cellranger_bin)
    if p.is_dir():
        p = p / "cellranger"
    if not p.exists():
        which = shutil.which(str(cellranger_bin))
        if which:
            p = Path(which)
    if not p.exists():
        raise FileNotFoundError(f"cellranger not found at: {cellranger_bin}")
    if not os.access(str(p), os.X_OK):
        raise PermissionError(f"cellranger is not executable: {p}")
    return p.resolve()


def _threads_from_cpu(eighty_percent: bool = True, override: int | None = None) -> int:
    if override is not None and override > 0:
        return int(override)
    cpu = os.cpu_count() or 1
    if not eighty_percent:
        return max(1, cpu)
    # floor to at least 1
    t = int(cpu * 0.8)
    return t if t > 0 else 1


def _filtered_gtf_path_for(gtf_unzipped: Path) -> Path:
    """
    Place filtered GTF next to source, appending '.filtered' before the .gtf extension.
    """
    if gtf_unzipped.suffix == ".gtf":
        return gtf_unzipped.with_name(gtf_unzipped.stem + ".filtered.gtf")
    # fallback (rare): if extension isn't .gtf, still append
    return gtf_unzipped.with_name(gtf_unzipped.name + ".filtered.gtf")


def _is_complete_cellranger_ref(ref_dir: Path) -> bool:
    """
    Heuristic check that a mkref output looks complete:
    - ref_dir exists
    - reference.json exists and is non-empty
    - fasta/genome.fa exists and is non-empty
    - star/ exists and contains at least one file
    """
    if not ref_dir.is_dir():
        return False
    if not _exists_nonempty(ref_dir / "reference.json"):
        return False
    if not _exists_nonempty(ref_dir / "fasta" / "genome.fa"):
        return False
    star_dir = ref_dir / "star"
    try:
        has_star_files = star_dir.is_dir() and any(star_dir.iterdir())
    except Exception:
        has_star_files = False
    return has_star_files


# --------------------------- Job file helpers ---------------------------


def _job_file_for(mkref_outdir: Path) -> Path:
    """
    Path to the per-release job status payload.
    No extension by request.
    """
    return mkref_outdir / "job_exit_status_message"


def _serialize_result(result: MkrefResult) -> dict[str, Any]:
    def p2s(p: Path | None) -> str | None:
        return str(p) if p is not None else None

    return {
        "status": result.get("status"),
        "message": result.get("message"),
        "release_dir": p2s(result.get("release_dir")),
        "filtered_gtf": p2s(result.get("filtered_gtf")),
        "mkref_dir": p2s(result.get("mkref_dir")),
        "threads_used": result.get("threads_used"),
    }


def _deserialize_result(payload: Mapping[str, Any]) -> MkrefResult:
    def s2p(s: str | None) -> Path | None:
        return Path(s) if s else None

    out: MkrefResult = {
        "status": str(payload.get("status") or ""),
        "message": str(payload.get("message") or ""),
        "release_dir": s2p(payload.get("release_dir")),  # type: ignore[arg-type]
        "filtered_gtf": s2p(payload.get("filtered_gtf")),  # type: ignore[arg-type]
        "mkref_dir": s2p(payload.get("mkref_dir")),  # type: ignore[arg-type]
        "threads_used": int(payload.get("threads_used") or 0),
    }
    return out


def _read_job_result(job_file: Path) -> MkrefResult | None:
    try:
        if not job_file.exists() or job_file.stat().st_size == 0:
            return None
        payload = json.loads(job_file.read_text())
        return _deserialize_result(payload)
    except Exception:
        return None  # corrupted/unknown format -> treat as not "fine"


def _write_job_result(job_file: Path, result: MkrefResult) -> None:
    try:
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job_file.write_text(json.dumps(_serialize_result(result), indent=2))
    except Exception:
        # Non-fatal: do not crash the build if we cannot persist.
        pass


# --------------------------- Types ---------------------------


class MkrefResult(TypedDict, total=False):
    status: str  # "done", "skipped", or "error"
    message: str
    release_dir: Path | None
    filtered_gtf: Path | None
    mkref_dir: Path | None
    threads_used: int


# --------------------------- Main entrypoint ---------------------------


def build_cellranger_mkref(
    paths_by_release: Mapping[int, Mapping[str, Path | None]],
    cellranger_bin: Path | str,
    *,
    assembly_label: str = "GRCh38",
    use_threads: int | None = None,  # None => use 80% of available CPUs
    silent: bool = False,
) -> dict[int, MkrefResult]:
    """
    Build Cell Ranger references for each release using the unzipped FASTA & GTF
    produced by `download_references(...)`.

    Returns
    -------
    dict[int, MkrefResult]
        Per-release status and paths.
    """
    cellranger = _resolve_cellranger(cellranger_bin)
    threads = _threads_from_cpu(eighty_percent=True, override=use_threads)

    results: dict[int, MkrefResult] = {}
    success_count = 0  # print at the end

    # Process releases in ascending order for reproducible logs
    for release in sorted(paths_by_release.keys()):
        rel_info = paths_by_release[release] or {}
        gtf_plain = rel_info.get("gtf")
        fasta_plain = rel_info.get("fasta")

        # Best-effort to resolve release_dir early (needed to locate job file)
        release_dir: Path | None = None
        if gtf_plain:
            release_dir = Path(gtf_plain).parent
        elif fasta_plain:
            release_dir = Path(fasta_plain).parent

        ref_name = f"reference_{assembly_label}_{release}"
        mkref_outdir: Path | None = (release_dir / ref_name) if release_dir else None
        job_file: Path | None = _job_file_for(mkref_outdir) if mkref_outdir else None

        # --------------- Early restore/skip if a "fine" job file exists ---------------
        if job_file and job_file.exists():
            prior = _read_job_result(job_file)
            if prior is not None:
                _log(
                    silent, f"[release {release}] Skipping (job_exit_status_message present) — restoring prior result."
                )
                results[release] = prior
                if prior.get("status") == "done":
                    success_count += 1
                continue  # do not redo work

        # --------------- Validate inputs (but still persist a job file for reproducibility) ---------------
        if not _exists_nonempty(gtf_plain) or not _exists_nonempty(fasta_plain):
            msg = (
                f"[release {release}] Missing unzipped files; "
                f"GTF present={_exists_nonempty(gtf_plain)}, FASTA present={_exists_nonempty(fasta_plain)}. "
                f"Skipping."
            )
            _log(silent, msg)
            filtered_gtf: Path | None = None
            if gtf_plain:
                try:
                    filtered_gtf = _filtered_gtf_path_for(Path(gtf_plain))  # type: ignore[arg-type]
                except Exception:
                    filtered_gtf = None

            result: MkrefResult = {
                "status": "error",
                "message": msg,
                "release_dir": release_dir,
                "filtered_gtf": filtered_gtf,
                "mkref_dir": mkref_outdir,
                "threads_used": threads,
            }
            if job_file:
                _write_job_result(job_file, result)
            results[release] = result
            # Do not count as success
            continue

        # From here on, both inputs exist and are non-empty
        gtf_plain = Path(gtf_plain)  # type: ignore[arg-type]
        fasta_plain = Path(fasta_plain)  # type: ignore[arg-type]
        release_dir = gtf_plain.parent  # by construction they live in GRCh38_<release>
        mkref_outdir = release_dir / ref_name
        job_file = _job_file_for(mkref_outdir)

        # --------------- Step 1: mkgtf -> protein_coding only ---------------
        filtered_gtf = _filtered_gtf_path_for(gtf_plain)

        if _exists_nonempty(filtered_gtf):
            _log(silent, f"[release {release}] Skipping mkgtf (already exists): {filtered_gtf.name}")
            mkgtf_done = True
        else:
            cmd_mkgtf = [
                str(cellranger),
                "mkgtf",
                str(gtf_plain),
                str(filtered_gtf),
                "--attribute=gene_biotype:protein_coding",
                "--attribute=gene_type:protein_coding",
            ]
            _log(silent, f"[release {release}] Running: {' '.join(cmd_mkgtf)}")
            proc_mkgtf = _run(cmd_mkgtf, cwd=release_dir)
            mkgtf_done = proc_mkgtf.returncode == 0 and _exists_nonempty(filtered_gtf)
            if not mkgtf_done:
                stderr = (proc_mkgtf.stderr or proc_mkgtf.stdout or "").strip()
                msg = f"[release {release}] mkgtf FAILED — {stderr or 'unknown error'}. Skipping."
                _log(silent, msg)
                result: MkrefResult = {
                    "status": "error",
                    "message": msg,
                    "release_dir": release_dir,
                    "filtered_gtf": filtered_gtf if filtered_gtf.exists() else None,
                    "mkref_dir": None,
                    "threads_used": threads,
                }
                _write_job_result(job_file, result)
                results[release] = result
                continue

            _log(silent, f"[release {release}] mkgtf OK: {filtered_gtf.name}")

        # --------------- Step 2: mkref ----- ----------
        if _is_complete_cellranger_ref(mkref_outdir):
            msg = f"[release {release}] Skipping mkref (complete ref exists): {mkref_outdir.name}"
            _log(silent, msg)
            result: MkrefResult = {
                "status": "skipped",
                "message": msg,
                "release_dir": release_dir,
                "filtered_gtf": filtered_gtf,
                "mkref_dir": mkref_outdir,
                "threads_used": threads,
            }
            # Persist so future runs restore identically
            _write_job_result(job_file, result)
            results[release] = result
            # Not a "done" success for this run
            continue

        cmd_mkref = [
            str(cellranger),
            "mkref",
            f"--genome={ref_name}",
            f"--fasta={str(fasta_plain)}",
            f"--genes={str(filtered_gtf)}",
            f"--nthreads={threads}",
        ]
        _log(silent, f"[release {release}] Running: {' '.join(cmd_mkref)} (cwd={release_dir})")
        proc_mkref = _run(cmd_mkref, cwd=release_dir)

        if proc_mkref.returncode != 0 or not _is_complete_cellranger_ref(mkref_outdir):
            stderr = (proc_mkref.stderr or proc_mkref.stdout or "").strip()
            msg = f"[release {release}] mkref FAILED — {stderr or 'unknown error'}"
            _log(silent, msg)
            result: MkrefResult = {
                "status": "error",
                "message": msg,
                "release_dir": release_dir,
                "filtered_gtf": filtered_gtf,
                "mkref_dir": mkref_outdir if mkref_outdir.exists() else None,
                "threads_used": threads,
            }
            _write_job_result(job_file, result)
            results[release] = result
            continue

        msg = f"[release {release}] mkref OK: {mkref_outdir.name}"
        _log(silent, msg)
        result: MkrefResult = {
            "status": "done",
            "message": msg,
            "release_dir": release_dir,
            "filtered_gtf": filtered_gtf,
            "mkref_dir": mkref_outdir,
            "threads_used": threads,
        }
        _write_job_result(job_file, result)
        results[release] = result
        success_count += 1

    # --------------- Final summary print ---------------
    _log(silent, f"{success_count}/{len(results)} releases ended successfully.")

    return results
