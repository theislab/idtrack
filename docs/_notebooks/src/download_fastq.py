#!/usr/bin/env python3

"""
Download FASTQ archives and associated auxiliary files (e.g., CSVs) for
10x Genomics (or similar) public datasets using curl, with checksum verification.

Key behaviors updated per request:
- If the final file already exists, we SKIP without computing MD5.
- Print "Downloading ..." (or "Resuming ...") only when we actually download.

Usage
-----
from fastq_downloader import download_fastq
download_fastq(fastq_datasets)  # will prompt for working directory
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---- Config defaults ---------------------------------------------------------

_DEFAULT_RETRIES = 3
_DEFAULT_RETRY_DELAY = 5  # seconds
_DEFAULT_RESUME = True
_DEFAULT_CURL = "curl"
_MD5_CHUNK = 16 * 1024 * 1024  # 16 MiB


# ---- Helpers ----------------------------------------------------------------


def _log(silent: bool, msg: str) -> None:
    if not silent:
        print(msg)


def _ensure_curl_available(curl_bin: str) -> None:
    """Raise RuntimeError if curl is not found/usable."""
    if shutil.which(curl_bin) is None:
        raise RuntimeError(f"'{curl_bin}' not found on PATH. Please install curl or provide curl_bin.")


def _md5_file(path: Path) -> str:
    """Compute MD5 checksum of a file in chunks. FIPS-safe fallback."""
    try:
        h = hashlib.md5()
    except Exception:
        try:
            h = hashlib.new("md5", usedforsecurity=False)  # type: ignore[call-arg]
        except Exception as e:
            raise RuntimeError(f"MD5 unavailable in this environment: {e}")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_MD5_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _curl_download(
    url: str,
    dest: Path,  # <-- may be a .part path
    curl_bin: str,
    retries: int,
    retry_delay: int,
    resume: bool,
) -> tuple[bool, str]:
    """
    Download 'url' to 'dest' using curl with retry and optional resume (-C -).
    Returns (ok, message).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    args = [
        curl_bin,
        "-fL",  # fail on HTTP errors; follow redirects
        "-sS",  # silent transfer but show errors
        "--retry",
        str(retries),
        "--retry-delay",
        str(retry_delay),
        "--retry-connrefused",
    ]

    # Only resume if there is something to resume.
    if resume and dest.exists() and dest.stat().st_size > 0:
        args += ["-C", "-"]

    args += ["-o", str(dest), url]

    cp = subprocess.run(args, capture_output=True, text=True)
    if cp.returncode == 0:
        return True, f"Downloaded: {dest.name}"
    else:
        return (
            False,
            f"curl failed (exit {cp.returncode}) for {url}\n{(cp.stderr or '').strip()}",
        )


def _download_one(
    file_spec: Mapping[str, Any],
    out_dir: Path,
    curl_bin: str,
    retries: int,
    retry_delay: int,
    resume: bool,
    silent: bool,
    checksum_retries: int = 1,
) -> dict[str, Any]:
    """
    Download a single file described by file_spec into out_dir.
    - If the FINAL file already exists, skip without doing MD5.
    - If we download (or resume), verify MD5 iff provided.
    """
    # Required fields
    if "url" not in file_spec or "filename" not in file_spec:
        raise ValueError("Each file spec must include 'url' and 'filename'.")

    url = str(file_spec["url"])
    filename = str(file_spec["filename"])
    expected_md5 = str(file_spec["md5"]) if file_spec.get("md5") else None

    dest = out_dir / filename
    part = dest.with_suffix(dest.suffix + ".part")

    # If final file already exists, SKIP without computing MD5
    if dest.exists():
        _log(silent, f"Skip (exists): {dest.name}")
        return {
            "file": str(dest),
            "url": url,
            "status": "skipped",
            "reason": "Already present; MD5 not checked by request.",
            "md5_ok": None,
        }

    attempts = 0
    while True:
        attempts += 1

        # Tell the truth about what we're doing
        if part.exists() and part.stat().st_size > 0 and resume:
            _log(silent, f"Resuming download: {filename}")
        else:
            _log(silent, f"Downloading: {filename}")

        ok, msg = _curl_download(
            url=url,
            dest=part,
            curl_bin=curl_bin,
            retries=retries,
            retry_delay=retry_delay,
            resume=resume,
        )
        if not ok:
            return {
                "file": str(dest),
                "url": url,
                "status": "error",
                "reason": msg,
                "md5_ok": False if expected_md5 else None,
            }

        # Verify checksum on the .part, then promote to final path.
        if expected_md5:
            try:
                actual = _md5_file(part)
            except FileNotFoundError:
                return {
                    "file": str(dest),
                    "url": url,
                    "status": "error",
                    "reason": "Temp file disappeared before checksum step.",
                    "md5_ok": False,
                }

            if actual.lower() == expected_md5.lower():
                try:
                    part.replace(dest)  # atomic promotion
                finally:
                    part.unlink(missing_ok=True)
                return {
                    "file": str(dest),
                    "url": url,
                    "status": "ok",
                    "reason": "Downloaded and verified." if attempts == 1 else "Verified after retry.",
                    "md5_ok": True,
                }
            else:
                # Clean up and maybe retry
                try:
                    part.unlink(missing_ok=True)
                except Exception:
                    pass

                if attempts <= (1 + checksum_retries):
                    _log(
                        silent,
                        f"Checksum mismatch for {dest.name} "
                        f"(expected {expected_md5}, got {actual}); retrying download...",
                    )
                    continue

                return {
                    "file": str(dest),
                    "url": url,
                    "status": "error",
                    "reason": f"MD5 mismatch after {attempts} attempt(s): expected {expected_md5}, got {actual}",
                    "md5_ok": False,
                }
        else:
            # No checksum provided: just promote .part -> final and return ok.
            try:
                part.replace(dest)
            finally:
                part.unlink(missing_ok=True)
            return {
                "file": str(dest),
                "url": url,
                "status": "ok",
                "reason": "Downloaded (no checksum provided).",
                "md5_ok": None,
            }


# ---- Public API --------------------------------------------------------------


def download_fastq(
    datasets: Mapping[str, Mapping[str, Any]],
    working_dir: str | Path | None = None,
    *,
    curl_bin: str = _DEFAULT_CURL,
    retries: int = _DEFAULT_RETRIES,
    retry_delay: int = _DEFAULT_RETRY_DELAY,
    resume: bool = _DEFAULT_RESUME,
    silent: bool = False,
) -> dict[str, Any]:
    """
    Master function to download FASTQ archives and associated aux files.

    Notes:
    - If the destination file already exists, we skip WITHOUT MD5 verification (per request).
    - For newly downloaded/resumed files, we verify MD5 if provided.
    """
    _ensure_curl_available(curl_bin)

    if working_dir is None:
        user_input = input("Enter working directory for downloads: ").strip()
        if not user_input:
            raise ValueError("Working directory is required.")
        working_dir = user_input

    root = Path(working_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    overall: dict[str, Any] = {}
    total = len(datasets)
    completed_ok = 0

    for ds_key, ds_spec_any in datasets.items():
        out_dir = root / ds_key
        out_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(ds_spec_any, Mapping):
            raise ValueError(f"Dataset '{ds_key}' is not a mapping.")
        ds_spec = ds_spec_any

        if "fastqs" not in ds_spec or not isinstance(ds_spec["fastqs"], Mapping):
            raise ValueError(f"Dataset '{ds_key}' must have a 'fastqs' mapping.")

        _log(silent, f"Processing dataset: {ds_key!r}")
        ds_result: MutableMapping[str, Any] = {"dir": str(out_dir), "files": {}}

        # Primary FASTQ archive
        fastqs_spec = ds_spec["fastqs"]
        try:
            fastqs_res = _download_one(
                file_spec=fastqs_spec,
                out_dir=out_dir,
                curl_bin=curl_bin,
                retries=retries,
                retry_delay=retry_delay,
                resume=resume,
                silent=silent,
            )
        except Exception as e:
            fastqs_res = {
                "file": str(out_dir / str(fastqs_spec.get("filename", "unknown"))),
                "url": str(fastqs_spec.get("url", "")),
                "status": "error",
                "reason": f"Exception: {e}",
                "md5_ok": False,
            }
        ds_result["files"]["fastqs"] = fastqs_res

        # Auxiliary files (e.g., CSV)
        aux_results: dict[str, Any] = {}
        aux_spec = ds_spec.get("aux")
        if isinstance(aux_spec, Mapping):
            for aux_name, aux_file_spec in aux_spec.items():
                if not isinstance(aux_file_spec, Mapping):
                    aux_results[aux_name] = {
                        "status": "error",
                        "reason": "Aux spec is not a mapping.",
                    }
                    continue
                try:
                    aux_results[aux_name] = _download_one(
                        file_spec=aux_file_spec,
                        out_dir=out_dir,
                        curl_bin=curl_bin,
                        retries=retries,
                        retry_delay=retry_delay,
                        resume=resume,
                        silent=silent,
                    )
                except Exception as e:
                    aux_results[aux_name] = {
                        "file": str(out_dir / str(aux_file_spec.get("filename", "unknown"))),
                        "url": str(aux_file_spec.get("url", "")),
                        "status": "error",
                        "reason": f"Exception: {e}",
                        "md5_ok": False,
                    }
        if aux_results:
            ds_result["files"]["aux"] = aux_results

        # Dataset-level status: 'ok' if all files are ok OR skipped
        def _is_ok(rec: Any) -> bool:
            return isinstance(rec, Mapping) and rec.get("status") in {"ok", "skipped"}

        ok_fastq = _is_ok(fastqs_res)
        ok_aux = all(_is_ok(v) for v in aux_results.values()) if aux_results else True

        if ok_fastq and ok_aux:
            ds_result["status"] = "ok"
            completed_ok += 1
        elif ok_fastq or ok_aux:
            ds_result["status"] = "partial"
        else:
            ds_result["status"] = "error"

        overall[ds_key] = ds_result

    _log(silent, f"- All done. Fully successful (ok or skipped): {completed_ok}/{total} dataset(s).")
    return overall
