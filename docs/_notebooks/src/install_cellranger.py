#!/usr/bin/env python3

"""Install a specific Cell Ranger version without sudo (reproducibly).

The main entry point is :func:`install_cellranger`, which you can call from a Jupyter notebook.
It downloads a versioned tarball to a user-writable tmp directory, verifies its MD5 (via both Python
and the ``md5sum`` CLI when available), and extracts the archive into a user-specified apps directory.
It **skips all steps** if the target version already exists at the destination.

Example:
    >>> from pathlib import Path
    >>> from install_cellranger import install_cellranger
    >>> cellranger_bin = install_cellranger(
    ...     VERSION="9.0.1",
    ...     EXPECTED_MD5="2efec98bff01f7a59edaf43724fae13f",
    ...     URL="https://cf.10xgenomics.com/releases/cell-exp/cellranger-9.0.1.tar.gz?Expires=blabla",
    ...     APPS_DIR=Path("/home/icb/kemal.inecik/tools/apps").expanduser(),
    ...     TMP_DIR=Path("/home/icb/kemal.inecik/tools/tmp").expanduser(),
    ... )
    >>> # Optionally make it available in *this* Python session:
    >>> import os
    >>> os.environ["PATH"] = f"{cellranger_bin.parent}:{os.environ['PATH']}"

Notes:
    * No sudo is required. Everything happens in user-writable paths you provide.
    * The downloaded ``*.tar.gz`` is kept in ``TMP_DIR`` for reproducibility/auditing.
    * Uses ``curl``, ``md5sum``, and ``tar`` if available; otherwise falls back to pure Python where possible.
    * Returns the full path to the ``cellranger`` launcher script inside the installed directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional


class InstallError(RuntimeError):
    """Raised when installation cannot proceed safely."""


def _print(msg: str) -> None:
    print(f"[cellranger] {msg}")


def _run(cmd: list[str], *, check: bool = True, capture_output: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    """Run a command via subprocess with sane defaults and helpful error messages."""
    try:
        return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        msg = f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        raise InstallError(msg) from e


def _python_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _cli_md5(path: Path) -> Optional[str]:
    """Return MD5 via the `md5sum` CLI if available, else None."""
    exe = shutil.which("md5sum")
    if not exe:
        return None
    res = _run([exe, str(path)], capture_output=True, text=True)
    # Output looks like: "2efec98bff01f7a59edaf43724fae13f  filename.tar.gz"
    return (res.stdout.split()[0] if res.stdout else "").strip() or None


def _safe_extract_tar_gz(archive: Path, dest_dir: Path) -> None:
    """Safely extract a .tar.gz using Python's tarfile (when `tar` is unavailable)."""
    def is_within_directory(directory: Path, target: Path) -> bool:
        try:
            directory = directory.resolve(strict=False)
            target = target.resolve(strict=False)
        except FileNotFoundError:
            # resolve(strict=False) avoids raising on missing paths, but handle just in case
            directory = directory.resolve()
            target = target.resolve()
        return str(target).startswith(str(directory))

    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf.getmembers():
            target_path = dest_dir / member.name
            if not is_within_directory(dest_dir, target_path):
                raise InstallError(f"Blocked path traversal attempt in archive: {member.name}")
        tf.extractall(dest_dir)


def install_cellranger(version: str, expected_md5: str, url: str, apps_dir: Path, tmp_dir: Path) -> Path:
    """Install a specific Cell Ranger version if missing, verifying MD5 and avoiding sudo.

    Args:
        version: Exact version string (e.g., ``"9.0.1"``). Used to form the install directory name.
        expected_md5: Expected MD5 checksum of the tarball, as a lowercase hex string.
        url: Full download URL to the versioned ``.tar.gz`` (usually a signed 10x Genomics link).
        apps_dir: Directory where the versioned folder (``cellranger-<VERSION>``) will be extracted.
        tmp_dir: Directory where the tarball will be downloaded and kept.

    Returns:
        Path to the ``cellranger`` launcher inside ``APPS_DIR / f"cellranger-{VERSION}"``.

    Raises:
        InstallError: If commands fail or checksums do not match.
    """
    # Normalize and ensure directories exist
    apps_dir = Path(apps_dir).expanduser().resolve()
    tmp_dir = Path(tmp_dir).expanduser().resolve()
    apps_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dest_dir = apps_dir / f"cellranger-{version}"
    launcher = dest_dir / "cellranger"
    tar_name = f"cellranger-{version}.tar.gz"
    tar_path = tmp_dir / tar_name

    # Fast path: already installed
    if launcher.exists():
        _print(f"Found existing install: {launcher}")
        return launcher

    # Download (skip if present)
    if not tar_path.exists():
        curl = shutil.which("curl")
        if curl:
            _print(f"Downloading with curl to {tar_path} ...")
            _run([curl, "-L", "-o", str(tar_path), url], check=True, capture_output=True)
        else:
            # Fallback: pure Python download
            import urllib.request
            _print(f"curl not found; downloading via urllib to {tar_path} ...")
            with urllib.request.urlopen(url) as resp, tar_path.open("wb") as out:
                shutil.copyfileobj(resp, out)
    else:
        _print(f"Reusing existing tarball: {tar_path}")

    # Verify MD5 (Python)
    py_md5 = _python_md5(tar_path)
    _print(f"MD5 (python) = {py_md5}")
    if py_md5.lower() != expected_md5.lower():
        # Try one re-download if possible
        curl = shutil.which("curl")
        if curl:
            _print("MD5 mismatch — re-downloading once with curl ...")
            _run([curl, "-L", "-o", str(tar_path), url], check=True, capture_output=True)
            py_md5 = _python_md5(tar_path)
            _print(f"MD5 after re-download (python) = {py_md5}")
        # Final decision
        if py_md5.lower() != expected_md5.lower():
            raise InstallError(
                f"Checksum mismatch. Expected {expected_md5} but got {py_md5}. "
                "The URL may be expired or incorrect."
            )

    # Verify MD5 (CLI md5sum) if available
    cli_md5 = _cli_md5(tar_path)
    if cli_md5:
        _print(f"MD5 (md5sum) = {cli_md5}")
        if cli_md5.lower() != expected_md5.lower():
            raise InstallError(
                f"md5sum reported {cli_md5}, but expected {expected_md5}. Aborting for safety."
            )
    else:
        _print("Warning: 'md5sum' not found; relied on Python MD5 only.")

    # Extract to APPS_DIR
    if dest_dir.exists():
        _print(f"Skipping extraction; destination already exists: {dest_dir}")
    else:
        tar_exe = shutil.which("tar")
        if tar_exe:
            _print(f"Extracting with tar to {apps_dir} ...")
            _run([tar_exe, "-xzf", str(tar_path), "-C", str(apps_dir)], check=True, capture_output=True)
        else:
            _print("tar not found; extracting via Python's tarfile (symlinks preserved).")
            _safe_extract_tar_gz(tar_path, apps_dir)

    # Sanity checks
    if not launcher.exists():
        raise InstallError(f"Install seems incomplete. Expected launcher missing: {launcher}")
    if not os.access(launcher, os.X_OK):
        # Try to make it executable (normally preserved by tar)
        try:
            current_mode = os.stat(launcher).st_mode
            os.chmod(launcher, current_mode | 0o111)
        except Exception as e:  # pragma: no cover
            raise InstallError(f"Launcher exists but is not executable: {launcher}") from e

    _print(f"Done. Installed at: {dest_dir}")
    _print(f"Tarball kept at: {tar_path}")
    return launcher
