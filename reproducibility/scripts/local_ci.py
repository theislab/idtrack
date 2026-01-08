#!/usr/bin/env python3
"""Local CI runner that mirrors the GitHub Actions workflows.

Why this exists
---------------
GitHub Actions is the source of truth for this repository's CI logic:

- `.github/workflows/build_package.yml` (packaging build + metadata checks)
- `.github/workflows/dependency_compatibility.yml` (dependency resolver matrix)

This script lets you run (a macOS-friendly approximation of) those workflows
locally *before pushing*. It also deliberately parses the matrices directly
from the workflow YAML, so you do not maintain the matrix in two places.

What it does
------------
Subcommands:

`build`
    Local equivalent of `build_package.yml`:

    1) Create an isolated venv under `.local-ci/build_package/venv`.
    2) Build sdist + wheel via `python -m build` (PEP 517).
    3) Validate metadata/rendering via `python -m twine check --strict`.
    4) Install the wheel with `--no-deps` and print the installed version via
       `importlib.metadata` (avoids importing runtime deps).
    5) Run a "wheel install smoke test" across the Python versions listed in
       `build_package.yml` that are available on your machine (parallelizable).

`compat`
    Local equivalent of `dependency_compatibility.yml`:

    1) Build a wheel once (like the `build-wheel` job).
    2) For each matrix entry in the workflow:
       - Create an isolated venv (one per case).
       - Write the workflow's multi-line constraints to a real constraints file.
       - Install the wheel under those constraints (`pip install -c ...`).
       - Run `pip check` to detect resolver conflicts/broken requirements.
       - Import `idtrack` and print `idtrack.__version__`.
    3) Each case writes a dedicated log under `.local-ci/.../logs/` and runs
       in parallel using a thread pool (`--jobs`).

`clean`
    Delete the local CI work directory (default: `.local-ci`).

Notes / differences from GitHub Actions
---------------------------------------
- This runs on *your* OS (often macOS), not Ubuntu runners. Some "legacy"
  constraints are harder on macOS arm64 because binary wheels may not exist.
  Example: `PyYAML<6` often falls back to building from source and can fail
  under build isolation; the script includes a targeted macOS workaround for
  those cases (install Cython + disable build isolation for that install).
- If you do not have a given interpreter available locally (e.g. `python3.12`)
  then the corresponding matrix entries are skipped with a note. Install
  interpreters via your preferred method (pyenv, conda, system python, etc).

Where artifacts go
------------------
Default work directory: `<repo>/.local-ci` (override with `--workdir`).

- Build artifacts: `.local-ci/build_package/dist/` and
  `.local-ci/dependency_compatibility/dist/`
- Virtualenvs: `.local-ci/**/venv*`
- Logs: `.local-ci/**/logs/*.log`

Typical usage
-------------
- Run everything: `python reproducibility/scripts/local_ci.py all`
- Clean: `python reproducibility/scripts/local_ci.py clean --yes`
- Build only: `python reproducibility/scripts/local_ci.py build`
- Compat only: `python reproducibility/scripts/local_ci.py compat --jobs 6`
- Narrow compat: `python reproducibility/scripts/local_ci.py compat --select 'pandas>=2' --pythons 3.10`

Implementation note
-------------------
This intentionally avoids adding a runtime dependency on PyYAML. The workflow
files are parsed with lightweight regex/state-machine logic tailored to the
fixed shapes used in this repository's Actions.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess  # noqa: S404
import sys
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PythonInterp:
    """Resolved Python interpreter for local CI runs."""

    version: str
    executable: str


@dataclass(frozen=True)
class CompatCase:
    """One dependency-compatibility matrix entry."""

    python_version: str
    name: str
    constraints: tuple[str, ...]


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / ".github" / "workflows").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from: {start}")


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env_merged = os.environ.copy()
    if env:
        env_merged.update(env)

    if stdout_path is None:
        print(f"+ {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, cwd=str(cwd), env=env_merged, text=True, check=check)  # noqa: S603

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if stdout_path.exists() else "w"
    with stdout_path.open(mode, encoding="utf-8") as fh:
        fh.write(f"+ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(  # noqa: S603
            cmd,
            cwd=str(cwd),
            env=env_merged,
            text=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=check,
        )


def _python_version(python_exe: str) -> str | None:
    try:
        out = subprocess.check_output(  # noqa: S603
            [python_exe, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"]
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.decode().strip()


def _find_python(version: str) -> PythonInterp | None:
    candidates = [
        f"python{version}",
        f"python3.{version.split('.', 1)[1]}" if version.startswith("3.") else None,
    ]
    for candidate in [c for c in candidates if c]:
        path = shutil.which(candidate)
        if not path:
            continue
        actual = _python_version(path)
        if actual == version:
            return PythonInterp(version=version, executable=path)

    # Allow using pyenv-managed interpreters if available.
    pyenv = shutil.which("pyenv")
    if pyenv:
        env = os.environ.copy()
        env["PYENV_VERSION"] = version
        try:
            pyenv_python = subprocess.check_output([pyenv, "which", "python"], env=env, text=True).strip()  # noqa: S603
        except (OSError, subprocess.CalledProcessError):
            pyenv_python = ""
        if pyenv_python and _python_version(pyenv_python) == version:
            return PythonInterp(version=version, executable=pyenv_python)

    return None


def _venv_python(venv_dir: Path) -> str:
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _venv_bin(venv_dir: Path, exe: str) -> str:
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / f"{exe}.exe")
    return str(venv_dir / "bin" / exe)


def _create_venv(venv_dir: Path, python_exe: str, *, recreate: bool) -> None:
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if venv_dir.exists():
        expected = _python_version(python_exe)
        actual = _python_version(_venv_python(venv_dir))
        if expected and actual and expected != actual:
            shutil.rmtree(venv_dir)
        else:
            return
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python_exe, "-m", "venv", str(venv_dir)], check=True)  # noqa: S603


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latest_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        raise FileNotFoundError(f"No wheels found in {dist_dir}")
    return wheels[0]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_yaml_inline_list(value: str) -> list[str]:
    items: list[str] = []
    for raw in value.split(","):
        cleaned = raw.strip()
        if not cleaned:
            continue
        items.append(_strip_quotes(cleaned))
    return items


def _parse_python_versions_from_build_workflow(workflow_path: Path) -> list[str]:
    lines = workflow_path.read_text(encoding="utf-8").splitlines()

    inline_re = re.compile(r"^\s*python-version:\s*\[(.*?)\]\s*$")
    for line in lines:
        if "${{" in line:
            continue
        m = inline_re.match(line)
        if not m:
            continue
        versions = [v for v in _parse_yaml_inline_list(m.group(1)) if v]
        if versions:
            return versions

    key_re = re.compile(r"^\s*python-version:\s*$")
    item_re = re.compile(r"^\s*-\s*(.+?)\s*$")
    for idx, line in enumerate(lines):
        if "${{" in line:
            continue
        if not key_re.match(line):
            continue
        base_indent = len(line) - len(line.lstrip(" "))
        versions: list[str] = []
        for next_line in lines[idx + 1 :]:
            if next_line.strip() == "" or next_line.lstrip().startswith("#"):
                continue
            indent = len(next_line) - len(next_line.lstrip(" "))
            if indent <= base_indent:
                break
            m_item = item_re.match(next_line)
            if not m_item:
                continue
            value = _strip_quotes(m_item.group(1))
            if "${{" in value:
                continue
            versions.append(value)
        if versions:
            return versions

    raise ValueError("Could not parse python-version matrix from workflow.")


def _parse_build_python_from_workflow(workflow_path: Path) -> str | None:
    version_re = re.compile(r'^\s*python-version:\s*["\']?(\d+\.\d+)["\']?\s*$')
    for line in workflow_path.read_text(encoding="utf-8").splitlines():
        if "${{" in line:
            continue
        m = version_re.match(line)
        if m:
            return m.group(1)
    return None


def _parse_compat_cases_from_workflow(workflow_path: Path) -> list[CompatCase]:
    text = workflow_path.read_text(encoding="utf-8").splitlines()

    cases: list[CompatCase] = []
    current_py: str | None = None
    current_name: str | None = None
    current_constraints: list[str] = []
    in_constraints = False
    constraints_indent: int | None = None

    entry_re = re.compile(r'^\s*-\s*python-version:\s*["\']?([^"\']+)["\']?\s*$')
    name_re = re.compile(r'^\s*dependency-set:\s*["\']?(.+?)["\']?\s*$')
    constraints_start_re = re.compile(r"^\s*constraints:\s*\|\s*$")

    def flush() -> None:
        nonlocal current_py, current_name, current_constraints, in_constraints, constraints_indent
        if current_py and current_name and current_constraints:
            cases.append(
                CompatCase(
                    python_version=current_py,
                    name=current_name,
                    constraints=tuple(line.strip() for line in current_constraints if line.strip()),
                )
            )
        current_py = None
        current_name = None
        current_constraints = []
        in_constraints = False
        constraints_indent = None

    for line in text:
        m_entry = entry_re.match(line)
        if m_entry:
            flush()
            current_py = m_entry.group(1)
            continue

        if current_py is None:
            continue

        if in_constraints:
            if line.strip() == "":
                continue
            indent = len(line) - len(line.lstrip(" "))
            if constraints_indent is None:
                constraints_indent = indent
            if indent < (constraints_indent or 0):
                in_constraints = False
                constraints_indent = None
            else:
                current_constraints.append(line.strip())
                continue

        m_name = name_re.match(line)
        if m_name:
            current_name = m_name.group(1)
            continue

        if constraints_start_re.match(line):
            in_constraints = True
            constraints_indent = None
            current_constraints = []
            continue

    flush()
    return cases


def _build_package(
    *,
    repo: Path,
    workdir: Path,
    python_exe: str,
    recreate_venvs: bool,
) -> Path:
    outdir = workdir / "build_package" / "dist"
    venv_dir = workdir / "build_package" / "venv"
    if outdir.exists():
        shutil.rmtree(outdir)
    _create_venv(venv_dir, python_exe, recreate=recreate_venvs)

    vpy = _venv_python(venv_dir)
    _run([vpy, "-m", "pip", "install", "--upgrade", "pip"], cwd=repo)
    _run([vpy, "-m", "pip", "install", "--upgrade", "build", "twine"], cwd=repo)

    outdir.mkdir(parents=True, exist_ok=True)
    _run([vpy, "-m", "build", "--outdir", str(outdir)], cwd=repo)
    artifacts = sorted(outdir.glob("*"))
    if not artifacts:
        raise FileNotFoundError(f"No build artifacts found in {outdir}")
    _run([vpy, "-m", "twine", "check", "--strict", *[str(p) for p in artifacts]], cwd=repo)

    wheel = _latest_wheel(outdir)
    _run([vpy, "-m", "pip", "install", "--no-deps", str(wheel)], cwd=repo)
    _run([vpy, "-c", "import importlib.metadata as m; print(m.version('idtrack'))"], cwd=repo)

    return wheel


def _run_wheel_smoke_test(
    *,
    repo: Path,
    workdir: Path,
    wheel: Path,
    python_exe: str,
    python_version: str,
    recreate_venvs: bool,
    keep_venvs: bool,
) -> tuple[str, bool, Path]:
    venv_dir = workdir / "build_package" / "install-venvs" / f"py{python_version}"
    log_path = workdir / "build_package" / "logs" / f"install-py{python_version}.log"

    _create_venv(venv_dir, python_exe, recreate=recreate_venvs)
    vpy = _venv_python(venv_dir)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    ok = True
    try:
        _run([vpy, "-m", "pip", "install", "--upgrade", "pip"], cwd=repo, stdout_path=log_path)
        _run([vpy, "-m", "pip", "install", "--no-deps", str(wheel)], cwd=repo, stdout_path=log_path)
        _run([vpy, "-c", "import importlib.metadata as m; print(m.version('idtrack'))"], cwd=repo, stdout_path=log_path)
    except subprocess.CalledProcessError:
        ok = False

    if ok and not keep_venvs:
        shutil.rmtree(venv_dir, ignore_errors=True)

    return python_version, ok, log_path


def _build_wheel_for_compat(
    *,
    repo: Path,
    workdir: Path,
    python_exe: str,
    recreate_venvs: bool,
) -> Path:
    outdir = workdir / "dependency_compatibility" / "dist"
    venv_dir = workdir / "dependency_compatibility" / "build-venv"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _create_venv(venv_dir, python_exe, recreate=recreate_venvs)
    vpy = _venv_python(venv_dir)
    _run([vpy, "-m", "pip", "install", "--upgrade", "pip"], cwd=repo)
    _run([vpy, "-m", "pip", "install", "build"], cwd=repo)
    _run([vpy, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=repo)

    return _latest_wheel(outdir)


def _run_compat_case(
    case: CompatCase,
    *,
    repo: Path,
    workdir: Path,
    wheel: Path,
    python_exe: str,
    recreate_venvs: bool,
    keep_venvs: bool,
) -> tuple[CompatCase, bool, Path]:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", f"py{case.python_version}-{case.name}").strip("_")
    case_hash = hashlib.sha256(
        ("\n".join([case.python_version, case.name, *case.constraints])).encode("utf-8")
    ).hexdigest()[:10]
    max_prefix_len = 180 - (len(case_hash) + 1)
    if len(safe_name) > max_prefix_len:
        safe_name = safe_name[:max_prefix_len].rstrip("_")
    case_id = f"{safe_name}-{case_hash}"
    venv_dir = workdir / "dependency_compatibility" / "venvs" / case_id
    log_path = workdir / "dependency_compatibility" / "logs" / f"{case_id}.log"
    constraints_path = workdir / "dependency_compatibility" / "constraints" / f"{case_id}.txt"

    _create_venv(venv_dir, python_exe, recreate=recreate_venvs)
    vpy = _venv_python(venv_dir)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    _write_lines(constraints_path, case.constraints)

    ok = True
    try:
        _run([vpy, "-m", "pip", "install", "--upgrade", "pip"], cwd=repo, stdout_path=log_path)
        # PyYAML<6 often has no wheels on macOS arm64, and PyYAML 5.4.1 can fail under
        # build isolation with `AttributeError: cython_sources`. Work around by ensuring
        # Cython is present and disabling build isolation for this install.
        if sys.platform == "darwin" and any(re.match(r"(?i)^pyyaml\s*<\s*6\b", c.strip()) for c in case.constraints):
            _run(
                [vpy, "-m", "pip", "install", "--upgrade", "setuptools", "wheel", "Cython<3"],
                cwd=repo,
                stdout_path=log_path,
            )
            pip_install = [vpy, "-m", "pip", "install", "--no-build-isolation", "-c", str(constraints_path), str(wheel)]
        else:
            pip_install = [vpy, "-m", "pip", "install", "-c", str(constraints_path), str(wheel)]
        _run(
            pip_install,
            cwd=repo,
            stdout_path=log_path,
        )
        _run([vpy, "-m", "pip", "check"], cwd=repo, stdout_path=log_path)
        _run([vpy, "-c", "import idtrack; print(idtrack.__version__)"], cwd=repo, stdout_path=log_path)
    except subprocess.CalledProcessError:
        ok = False

    if ok and not keep_venvs:
        shutil.rmtree(venv_dir, ignore_errors=True)

    return case, ok, log_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local CI CLI and return an exit status."""
    parser = argparse.ArgumentParser(
        prog="local_ci.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=textwrap.dedent(
            """
            Examples:
              # Run both build + dependency compatibility checks
              python reproducibility/scripts/local_ci.py all

              # Only build-package (PEP 517 build + metadata check + wheel install)
              python reproducibility/scripts/local_ci.py build

              # Only dependency compatibility (constraints matrix)
              python reproducibility/scripts/local_ci.py compat --jobs 6

              # Run a single compat case (by regex)
              python reproducibility/scripts/local_ci.py compat --select 'pandas>=2 / numpy>=2'
            """
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=_repo_root() / ".local-ci",
        help="Directory for local CI artifacts/venvs (default: .local-ci).",
    )
    parser.add_argument(
        "--recreate-venvs",
        action="store_true",
        help="Recreate all virtualenvs (clean-room runs, slower).",
    )
    parser.add_argument(
        "--keep-venvs",
        action="store_true",
        help="Keep per-case virtualenvs even on success (useful for debugging).",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all", help="Run build + dependency compatibility checks.")

    clean_cmd = sub.add_parser("clean", help="Remove the local CI workdir (default: .local-ci).")
    clean_cmd.add_argument(
        "--yes",
        action="store_true",
        help="Delete without prompting.",
    )

    build_cmd = sub.add_parser("build", help="Run the build_package.yml equivalent locally.")
    build_cmd.add_argument(
        "--pythons",
        type=str,
        default="",
        help="Comma-separated Python versions for a wheel install smoke test (default: all supported versions found).",
    )
    build_cmd.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallelism for wheel install smoke tests.",
    )

    compat = sub.add_parser("compat", help="Run the dependency_compatibility.yml equivalent locally.")
    compat.add_argument(
        "--workflow",
        type=Path,
        default=_repo_root() / ".github" / "workflows" / "dependency_compatibility.yml",
        help="Path to dependency_compatibility.yml (default: repo workflow).",
    )
    compat.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel jobs for the constraints matrix.",
    )
    compat.add_argument(
        "--select",
        type=str,
        default="",
        help="Only run compat cases whose name matches this regex.",
    )
    compat.add_argument(
        "--pythons",
        type=str,
        default="",
        help="Comma-separated Python versions to run (e.g. 3.10,3.12).",
    )

    args = parser.parse_args(argv)
    repo = _repo_root()
    workdir: Path = args.workdir

    if args.cmd == "clean":
        if not workdir.exists():
            print(f"Nothing to clean (missing): {workdir}", flush=True)
            return 0
        if not args.yes:
            answer = input(f"Delete local CI workdir {workdir}? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("Canceled.", flush=True)
                return 1
        shutil.rmtree(workdir)
        print(f"Removed: {workdir}", flush=True)
        return 0

    workdir.mkdir(parents=True, exist_ok=True)

    compat_workflow = repo / ".github" / "workflows" / "dependency_compatibility.yml"
    try:
        preferred_build_python = _parse_build_python_from_workflow(compat_workflow) or "3.12"
    except FileNotFoundError:
        preferred_build_python = "3.12"

    current_python_version = _python_version(sys.executable)
    preferred_interp = _find_python(preferred_build_python)
    build_python = preferred_interp or (_find_python(current_python_version) if current_python_version else None)
    if build_python is None:
        build_python = PythonInterp(version=current_python_version or "<unknown>", executable=sys.executable)
    elif preferred_interp is None and preferred_build_python:
        print(
            f"==> note: workflow wheel-build uses py{preferred_build_python}; not found locally, using py{build_python.version}",
            file=sys.stderr,
            flush=True,
        )

    if args.cmd in {"build", "all"}:
        print(f"==> build: using {build_python.executable} (py{build_python.version})", flush=True)
        try:
            wheel = _build_package(
                repo=repo,
                workdir=workdir,
                python_exe=build_python.executable,
                recreate_venvs=args.recreate_venvs,
            )
        except subprocess.CalledProcessError as e:
            print(f"build failed: {e}", file=sys.stderr)
            return e.returncode

        requested = getattr(args, "pythons", "")
        if requested:
            versions = [v.strip() for v in requested.split(",") if v.strip()]
        else:
            build_workflow = repo / ".github" / "workflows" / "build_package.yml"
            try:
                versions = _parse_python_versions_from_build_workflow(build_workflow)
            except (FileNotFoundError, ValueError) as e:
                print(f"==> build: could not parse python versions from {build_workflow}: {e}", file=sys.stderr)
                versions = []

        resolved: list[PythonInterp] = []
        for v in versions:
            interp = _find_python(v)
            if interp is not None:
                resolved.append(interp)

        if not resolved:
            print("==> build: no additional Python interpreters found for wheel install smoke tests", flush=True)
        else:
            jobs = max(1, int(getattr(args, "jobs", 1)))
            print(f"==> build: wheel install smoke tests on {len(resolved)} interpreter(s) ({jobs} job(s))", flush=True)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            failures: list[tuple[str, Path]] = []
            successes = 0
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futures = [
                    ex.submit(
                        _run_wheel_smoke_test,
                        repo=repo,
                        workdir=workdir,
                        wheel=wheel,
                        python_exe=interp.executable,
                        python_version=interp.version,
                        recreate_venvs=args.recreate_venvs,
                        keep_venvs=args.keep_venvs,
                    )
                    for interp in resolved
                ]
                for fut in as_completed(futures):
                    ver, ok, log_path = fut.result()
                    if ok:
                        successes += 1
                        print(f"[OK]   build wheel install :: py{ver}", flush=True)
                    else:
                        failures.append((ver, log_path))
                        print(f"[FAIL] build wheel install :: py{ver}  (log: {log_path})", flush=True)

            print(f"==> build summary: {successes}/{len(resolved)} passed, {len(failures)} failed", flush=True)
            if failures:
                return 1

    if args.cmd in {"compat", "all"}:
        workflow_path: Path = getattr(args, "workflow", repo / ".github" / "workflows" / "dependency_compatibility.yml")
        cases = _parse_compat_cases_from_workflow(workflow_path)
        if getattr(args, "select", ""):
            pattern = re.compile(args.select)
            cases = [c for c in cases if pattern.search(c.name)]
        if getattr(args, "pythons", ""):
            allowed = {v.strip() for v in args.pythons.split(",") if v.strip()}
            cases = [c for c in cases if c.python_version in allowed]

        if not cases:
            print(f"No compat cases found in {workflow_path}", file=sys.stderr)
            return 2

        print(f"==> compat: building wheel once with {build_python.executable} (py{build_python.version})", flush=True)
        try:
            wheel = _build_wheel_for_compat(
                repo=repo,
                workdir=workdir,
                python_exe=build_python.executable,
                recreate_venvs=args.recreate_venvs,
            )
        except subprocess.CalledProcessError as e:
            print(f"wheel build failed: {e}", file=sys.stderr)
            return e.returncode

        jobs = max(1, int(getattr(args, "jobs", 1)))
        print(f"==> compat: running {len(cases)} case(s) with up to {jobs} parallel job(s)", flush=True)

        # Resolve python executables up-front and skip cases without an interpreter.
        resolved: list[tuple[CompatCase, PythonInterp]] = []
        skipped: list[CompatCase] = []
        python_cache: dict[str, PythonInterp | None] = {}
        for case in cases:
            if case.python_version not in python_cache:
                python_cache[case.python_version] = _find_python(case.python_version)
            interp = python_cache[case.python_version]
            if interp is None:
                skipped.append(case)
                continue
            resolved.append((case, interp))

        if skipped:
            print("Skipping cases with missing Python interpreters:")
            for case in skipped:
                print(f"  - py{case.python_version}: {case.name}", flush=True)

        failures: list[tuple[CompatCase, Path]] = []
        successes = 0

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [
                ex.submit(
                    _run_compat_case,
                    case,
                    repo=repo,
                    workdir=workdir,
                    wheel=wheel,
                    python_exe=interp.executable,
                    recreate_venvs=args.recreate_venvs,
                    keep_venvs=args.keep_venvs,
                )
                for case, interp in resolved
            ]
            for fut in as_completed(futures):
                case, ok, log_path = fut.result()
                if ok:
                    successes += 1
                    print(f"[OK]   py{case.python_version} :: {case.name}", flush=True)
                else:
                    failures.append((case, log_path))
                    print(f"[FAIL] py{case.python_version} :: {case.name}  (log: {log_path})", flush=True)

        total = len(resolved)
        print(f"==> compat summary: {successes}/{total} passed, {len(failures)} failed", flush=True)
        if failures:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
