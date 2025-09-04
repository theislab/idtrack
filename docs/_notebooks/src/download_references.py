#!/usr/bin/env python3

from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Tuple, Optional, TypedDict


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and capture output (stdout/stderr)."""
    return subprocess.run(cmd, capture_output=True, text=True)


def _log(silent: bool, msg: str) -> None:
    """Print only when silent=False."""
    if not silent:
        print(msg)


def _exists_nonempty(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _curl_download(url: str, dest: Path) -> Tuple[bool, str]:
    """
    Download URL to dest using curl with resume & retry.
    Returns (ok, message).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists and looks non-empty, skip
    if _exists_nonempty(dest):
        return True, f"Skipped (exists): {dest.name}"

    cmd = [
        "curl",
        "-fL",
        "--retry", "3",
        "--retry-delay", "2",
        "-C", "-",
        "-o", str(dest),
        url,
    ]
    proc = _run(cmd)
    if proc.returncode == 0:
        return True, f"Downloaded: {dest.name}"
    reason = (proc.stderr or proc.stdout).strip() or "curl error"
    return False, f"FAILED: {dest.name} — {reason}"


def _gtf_name_url(release: int) -> Tuple[str, str]:
    """Return (filename, url) for a given Ensembl release GTF."""
    name = f"Homo_sapiens.GRCh38.{release}.gtf.gz"
    url = f"https://ftp.ensembl.org/pub/release-{release}/gtf/homo_sapiens/{name}"
    return name, url


def _fasta_name_url(release: int, fasta_name: str) -> Tuple[str, str]:
    """Return (filename, url) for the primary assembly FASTA in a given release."""
    url = f"https://ftp.ensembl.org/pub/release-{release}/fasta/homo_sapiens/dna/{fasta_name}"
    return fasta_name, url


def _versioned_fasta_gz_name(release: int) -> str:
    """Target filename like Homo_sapiens.GRCh38.<release>.dna.primary_assembly.fa.gz."""
    return f"Homo_sapiens.GRCh38.{release}.dna.primary_assembly.fa.gz"


def _decompress_gzip(src_gz: Path, *, keep_gz: bool = True) -> Tuple[bool, str, Optional[Path]]:
    """
    Decompress <src_gz> to the same directory.
    If keep_gz is False, delete the .gz after a successful extraction.
    Returns (ok, message, dest_path_or_none).
    """
    if not src_gz.exists():
        return False, f"Missing for gunzip: {src_gz.name}", None

    dest = src_gz.with_suffix("")  # drop ".gz"
    try:
        # If already decompressed, honor keep_gz by optionally removing the .gz
        if _exists_nonempty(dest):
            msg = f"Skipped gunzip (exists): {dest.name}"
            if not keep_gz and src_gz.exists():
                try:
                    src_gz.unlink()
                    msg += " (deleted .gz)"
                except Exception as e:
                    msg += f" (failed to delete .gz: {e})"
            return True, msg, dest

        with gzip.open(src_gz, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)

        if not keep_gz:
            try:
                src_gz.unlink()
            except Exception as e:
                return True, f"Gunzipped: {dest.name} (failed to delete .gz: {e})", dest

        return True, f"Gunzipped: {dest.name}", dest
    except Exception as e:
        return False, f"FAILED gunzip {src_gz.name} — {e}", None


class ReleasePaths(TypedDict, total=False):
    gtf_gz: Optional[Path]
    gtf: Optional[Path]
    fasta_gz: Optional[Path]
    fasta: Optional[Path]


def download_references(
    workdir: Path,
    releases: Iterable[int],
    fasta_name: str,
    *,
    gunzip: bool = True,
    keep_gz: bool = True,
    silent: bool = False,
) -> dict[int, ReleasePaths]:
    """
    Download Ensembl GTFs for the specified releases and the GRCh38 primary assembly FASTA,
    rename FASTAs to include the release number, and (optionally) gunzip.

    Key behavior:
    - Per-release subfolder is now "GRCh38_<release>".
    - FASTA: before downloading original name (A), we first look for the versioned name (B)
      and/or its unzipped file and skip downloading if present.
    - GTF: skip downloading if the unzipped file already exists (even if .gz was deleted).

    Parameters
    ----------
    workdir : Path
        Root directory where per-release subfolders will be created.
    releases : Iterable[int]
        Sequence of Ensembl release numbers (e.g., range(80, 116)).
    fasta_name : str
        FASTA filename to fetch from each release's primary assembly path
        (e.g. 'Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz').
    gunzip : bool, default True
        If True, gunzip both GTF and FASTA files after download.
    keep_gz : bool, default True
        If True, keep the .gz files after successful extraction. Only relevant if gunzip=True.
    silent : bool, default False
        If True, suppress console logs (still returns paths).

    Returns
    -------
    dict[int, ReleasePaths]
        For each release, paths for gtf_gz, gtf (unzipped), fasta_gz (renamed, versioned),
        and fasta (unzipped). Any missing artifact is returned as None.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    results: dict[int, ReleasePaths] = {}

    for r in releases:
        rel_dir = workdir / f"GRCh38_{r}"
        rel_dir.mkdir(parents=True, exist_ok=True)

        # -------------------- GTF (download if neither gz nor unzipped exist) --------------------
        gtf_name, gtf_url = _gtf_name_url(r)
        gtf_gz_path = rel_dir / gtf_name
        gtf_path = gtf_gz_path.with_suffix("")

        if _exists_nonempty(gtf_gz_path) or _exists_nonempty(gtf_path):
            # Already have something usable; no fresh download
            _log(
                silent,
                f"[release {r}] Skipped GTF download (already present): "
                f"{gtf_gz_path.name if _exists_nonempty(gtf_gz_path) else gtf_path.name}"
            )
        else:
            ok, msg = _curl_download(gtf_url, gtf_gz_path)
            _log(silent, f"[release {r}] {msg}")

        # Gunzip GTF if requested
        gtf_unzipped: Optional[Path] = None
        if gunzip:
            if _exists_nonempty(gtf_gz_path):
                ok2, msg2, dest = _decompress_gzip(gtf_gz_path, keep_gz=keep_gz)
                _log(silent, f"[release {r}] {msg2}")
                if ok2 and dest:
                    gtf_unzipped = dest
            elif _exists_nonempty(gtf_path):
                gtf_unzipped = gtf_path
                _log(silent, f"[release {r}] GTF already unzipped: {gtf_path.name}")
        else:
            if _exists_nonempty(gtf_path):
                gtf_unzipped = gtf_path
            target_note = gtf_gz_path.name if gtf_gz_path.exists() else gtf_path.name
            _log(silent, f"[release {r}] Gunzip disabled; leaving {target_note} as-is.")

        # -------------------- FASTA (prefer versioned name B) --------------------
        src_name, fasta_url = _fasta_name_url(r, fasta_name)  # original A
        src_path = rel_dir / src_name
        dst_gz = rel_dir / _versioned_fasta_gz_name(r)        # versioned B
        dst_unzipped = dst_gz.with_suffix("")                  # B without .gz

        # If B (or its unzipped twin) already exists, do NOT try to download A
        if _exists_nonempty(dst_gz) or _exists_nonempty(dst_unzipped):
            which = dst_gz.name if _exists_nonempty(dst_gz) else dst_unzipped.name
            _log(silent, f"[release {r}] Skipped FASTA download (already present): {which}")
        elif _exists_nonempty(src_path):
            # Legacy A exists locally -> just rename to B
            src_path.rename(dst_gz)
            _log(silent, f"[release {r}] Renamed: {dst_gz.name}")
        else:
            # Download A and then rename to B
            ok, msg = _curl_download(fasta_url, src_path)
            _log(silent, f"[release {r}] {msg}")
            if _exists_nonempty(src_path):
                src_path.rename(dst_gz)
                _log(silent, f"[release {r}] Renamed: {dst_gz.name}")

        # Gunzip FASTA if requested
        fasta_unzipped: Optional[Path] = None
        if gunzip:
            if _exists_nonempty(dst_gz):
                ok3, msg3, dest = _decompress_gzip(dst_gz, keep_gz=keep_gz)
                _log(silent, f"[release {r}] {msg3}")
                if ok3 and dest:
                    fasta_unzipped = dest
            elif _exists_nonempty(dst_unzipped):
                fasta_unzipped = dst_unzipped
                _log(silent, f"[release {r}] FASTA already unzipped: {dst_unzipped.name}")
        else:
            if _exists_nonempty(dst_unzipped):
                fasta_unzipped = dst_unzipped
            target_note = dst_gz.name if dst_gz.exists() else dst_unzipped.name
            _log(silent, f"[release {r}] Gunzip disabled; leaving {target_note} as-is.")

        # -------------------- Collect results --------------------
        results[r] = {
            "gtf_gz": gtf_gz_path if _exists_nonempty(gtf_gz_path) else None,
            "gtf": (gtf_unzipped if (gtf_unzipped and _exists_nonempty(gtf_unzipped))
                    else (gtf_path if _exists_nonempty(gtf_path) else None)),
            "fasta_gz": dst_gz if _exists_nonempty(dst_gz) else None,
            "fasta": (fasta_unzipped if (fasta_unzipped and _exists_nonempty(fasta_unzipped))
                      else (dst_unzipped if _exists_nonempty(dst_unzipped) else None)),
        }

    return results
