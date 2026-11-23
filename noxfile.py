"""Nox sessions."""

import os
import shlex
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session, session
except ImportError:
    print("nox-poetry was not found in the environment of nox; install it next to nox: pip install nox-poetry")
    sys.exit(1)


def _ensure_poetry_export_available() -> None:
    """Fail fast with a helpful message if `poetry export` is unavailable.

    `nox-poetry` relies on `poetry export`, which is provided by the
    `poetry-plugin-export` plugin for newer Poetry versions.

    Raises:
        SystemExit: If Poetry is missing or `poetry export` is unavailable.
    """
    poetry = shutil.which("poetry")
    if poetry is None:
        print(
            "Poetry was not found on PATH.\n"
            "Install Poetry and the export plugin, e.g.\n"
            "  - pip:   pip install poetry poetry-plugin-export\n"
            "  - pipx:  pipx install poetry && pipx inject poetry poetry-plugin-export\n"
            "  - conda: conda install -c conda-forge poetry poetry-plugin-export\n"
        )
        raise SystemExit(1)

    proc = subprocess.run(  # noqa: S603
        [poetry, "export", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode == 0:
        return

    out = (proc.stdout or "").strip()
    if "command export does not exist" in out.lower():
        print(
            "Your Poetry installation does not provide `poetry export`.\n"
            "Install the export plugin into the same Poetry environment:\n"
            "  - pipx:  pipx inject poetry poetry-plugin-export\n"
            "  - poetry: poetry self add poetry-plugin-export\n"
            "  - pip/conda: ensure `poetry-plugin-export` is installed next to `poetry`\n"
        )
        raise SystemExit(1)

    print("Poetry failed to run `poetry export --help`.")
    if out:
        print(out)
    raise SystemExit(1)


_ensure_poetry_export_available()

package = "idtrack"
python_versions = ["3.12", "3.13", "3.14", "3.15"]
# The interpreter of the single-version sessions in CI: the oldest supported, so that type checks and doctests catch
# any use of a newer Python's features (see .github/workflows/run_tests.yml).
default_python = python_versions[0]
nox.options.sessions = (
    "pre-commit",
    "pip-audit",
    "mypy",
    "tests",
    "xdoctest",
    "docs-build",
)


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """
    assert session.bin is not None  # noqa: S101

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir for bindir in (repr(session.bin), shlex.quote(session.bin))
    ]

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not any(Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text for bindir in bindirs):
            continue

        lines = text.splitlines()

        for executable, header in headers.items():
            if executable in lines[0].lower():
                lines.insert(1, dedent(header))
                hook.write_text("\n".join(lines))
                break


@session(name="pre-commit", python=python_versions)
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files"]
    session.install(
        "black",
        "flake8",
        "flake8-bandit",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-builtins",
        "flake8-colors",
        "flake8-rst-docstrings",
        "isort",
        "bandit",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pyupgrade",
    )
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(name="pip-audit", python=default_python)
def pip_audit(session: Session) -> None:
    """Scan the locked dependencies of the package and its extras for known vulnerabilities."""
    # The lock is exported with hashes, so every requirement is pinned and pip-audit needs no resolver (--disable-pip).
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as reqs_file:
        reqs_path = Path(reqs_file.name)

    try:
        session.run(
            "poetry", "export", "--format=requirements.txt", "--all-extras", "--output", str(reqs_path), external=True
        )
        session.install("pip-audit")
        session.run("pip-audit", "--requirement", str(reqs_path), "--disable-pip", "--progress-spinner", "off")
    finally:
        if reqs_path.exists():
            reqs_path.unlink()


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["idtrack", "tests", "docs/conf.py"]
    session.install(".[harmonize]")
    session.install(
        "mypy",
        "pytest",
        "types-requests",
        "types-PyMySQL",
        "types-PyYAML",
    )
    session.run("mypy", *args)


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".[harmonize]")
    session.install("coverage[toml]", "pytest", "pygments")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            # Without the posargs of this session, which are pytest's.
            session.notify("coverage", posargs=[])


@session(python=default_python)
def coverage(session: Session) -> None:
    """Produce the coverage report: combine the data files and report, or run `coverage <posargs>` if given."""
    session.install("coverage[toml]")

    if session.posargs:
        session.run("coverage", *session.posargs)
        return

    if any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")
    session.run("coverage", "report", "-i")


@session(python=python_versions)
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(".[harmonize]")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install(".[harmonize]")
    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", package, *args)


@session(name="docs-build", python=python_versions)
def docs_build(session: Session) -> None:
    """Build the documentation, every warning an error (the tools of docs/requirements.txt, at the lock's versions)."""
    args = session.posargs or ["-W", "--keep-going", "docs", "docs/_build"]
    session.install(".[harmonize]")
    session.install("-r", "docs/requirements.txt")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions)
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".[harmonize]")
    session.install("-r", "docs/requirements.txt", "sphinx-autobuild")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
