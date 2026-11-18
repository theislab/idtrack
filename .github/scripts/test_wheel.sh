#!/usr/bin/env bash
# Install a built wheel of idtrack into a fresh virtual environment and run the unit suite against the installed
# package, not against the source tree.
#
# Usage: test_wheel.sh <wheel> [<extra>]      (e.g. test_wheel.sh dist/idtrack-0.1.0-py3-none-any.whl harmonize)
#
#   <wheel>   the wheel to test.
#   <extra>   an extra to install with it (`harmonize`), or nothing for the plain install. Without `harmonize`, the
#             tests that need anndata or scipy skip themselves; the script checks that neither is installed.
#
# Environment: PYTHON, the interpreter the virtual environment is made from (default `python3`); WORK_DIR, an empty
# or missing folder for the virtual environment and the copy of the tests (default: a new temporary folder).
#
# The unit tests and `pyproject.toml` (pytest reads its settings there) are copied into WORK_DIR, which holds no
# `idtrack` folder, and pytest runs there with `--import-mode=importlib`, so `import idtrack` can only find the
# installed wheel; the script fails if `idtrack.__file__` is not in the virtual environment's site-packages. pytest is
# installed at the version of poetry.lock. Used by the `test-wheel` job of .github/workflows/build_package.yml, and
# runnable as it is on any machine with the Python version to test.
set -euo pipefail

[ $# -ge 1 ] && [ $# -le 2 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
wheel="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
extra="${2:-}"
python="${PYTHON:-python3}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$wheel" ] || { echo "no such wheel: $wheel" >&2; exit 2; }

work="${WORK_DIR:-$(mktemp -d)}"
mkdir -p "$work"
[ -z "$(ls -A "$work")" ] || { echo "WORK_DIR is not empty: $work" >&2; exit 2; }
echo "wheel: $wheel"
echo "extra: ${extra:-(none)}"
echo "work folder: $work"

"$python" -m venv "$work/venv"
venv_python="$work/venv/bin/python"
[ -x "$venv_python" ] || venv_python="$work/venv/Scripts/python.exe"  # Windows layout
"$venv_python" -m pip install --upgrade pip

# The wheel (with the extra if one is given), then pytest at the version of the lock.
target="$wheel"
[ -z "$extra" ] || target="$wheel[$extra]"
"$venv_python" -m pip install "$target"
pytest_version="$("$venv_python" -c '
import sys, tomllib
lock = tomllib.load(open(sys.argv[1], "rb"))
print(next(p["version"] for p in lock["package"] if p["name"] == "pytest"))
' "$repo/poetry.lock")"
"$venv_python" -m pip install "pytest==$pytest_version"
"$venv_python" -m pip check

# The tests and the pytest settings, in a folder without the package's sources.
mkdir "$work/run"
cp -R "$repo/tests" "$work/run/tests"
cp "$repo/pyproject.toml" "$work/run/pyproject.toml"
cd "$work/run"

# Without `harmonize`, its dependencies must be absent, or this run would not test the plain install.
"$venv_python" - "$extra" <<'EOF'
import importlib.util
import sys

present = [name for name in ("anndata", "scipy") if importlib.util.find_spec(name) is not None]
print("anndata/scipy installed:", ", ".join(present) or "neither")
if sys.argv[1] != "harmonize" and present:
    sys.exit("the plain install must not hold anndata or scipy")
EOF

# idtrack must come from site-packages (the same sys.path as `python -m pytest` in this folder).
"$venv_python" - <<'EOF'
import pathlib
import sys
import sysconfig

import idtrack

where = pathlib.Path(idtrack.__file__).resolve()
site = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()
print("idtrack", getattr(idtrack, "__version__", "?"), "imported from", where)
if site not in where.parents:
    sys.exit(f"idtrack is not imported from the virtual environment's site-packages ({site})")
EOF

"$venv_python" -m pytest -p no:cacheprovider --import-mode=importlib tests/unit
