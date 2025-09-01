#!/usr/bin/env bash
# IDTrack developer menu (prints the invoked command in yellow; does NOT color command output)
# Usage: ./idtrack_dev_menu.sh

# --- constants ---------------------------------------------------------------
PKG_DIR="/Users/kemalinecik/git_nosync/idtrack"

# Colors (only for our own one-line echos; never color command output)
if [[ -t 1 ]]; then
  YELLOW=$'\033[33m'   # non-bold yellow
  GREEN=$'\033[32m'
  RED=$'\033[31m'
  RESET=$'\033[0m'
else
  YELLOW=""; GREEN=""; RED=""; RESET=""
fi

# --- safety checks -----------------------------------------------------------
if [[ ! -d "$PKG_DIR" ]]; then
  echo "Package directory not found at: $PKG_DIR"
  echo "Edit PKG_DIR at the top of this script."
  exit 1
fi

# --- conda setup -------------------------------------------------------------
init_conda() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    __conda_setup="$("$CONDA_EXE" 'shell.bash' 'hook' 2>/dev/null)" || true
    if [[ -n "$__conda_setup" ]]; then
      eval "$__conda_setup"
      return
    fi
  fi
  for p in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$p" ]]; then
      # shellcheck disable=SC1090
      . "$p"
      return
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda 'shell.bash' 'hook' 2>/dev/null)"
    return
  fi
  echo "Could not initialize conda. Run 'conda init bash' and open a new shell."
  exit 1
}

show_envs() {
  init_conda
  echo "Detected conda environments:"
  conda env list | sed '1,2d' || true
}

choose_env() {
  echo "Which conda environment should be used?"
  echo "Examples: idtrack_poetry_3_9_env, idtrack_poetry_11_env"
  show_envs
  read -rp "Enter env name [idtrack_poetry_3_9_env]: " _env
  ENV_NAME="${_env:-idtrack_poetry_3_9_env}"
  echo "Using conda env: $ENV_NAME"
}

# Print the full command we're about to run (in yellow), always from project root, and ending with deactivate + cd
_preview_cmd() {
  local cmd="$*"
  printf "%s%s%s\n" "$YELLOW" \
    "conda activate \"$ENV_NAME\" && cd \"$PKG_DIR\" && $cmd && conda deactivate && cd" \
    "$RESET"
}

# Run a command: activate env, cd to project root, run cmd, always deactivate, then cd (back to HOME)
run_in_env() {
  local cmd="$*"
  init_conda
  _preview_cmd "$cmd"

  conda activate "$ENV_NAME" || { printf "%s✖ Failed%s to activate: %s\n" "$RED" "$RESET" "$ENV_NAME"; return 1; }
  cd "$PKG_DIR" || { conda deactivate || true; printf "%s✖ Failed%s to cd to: %s\n" "$RED" "$RESET" "$PKG_DIR"; return 1; }

  bash -lc "$cmd"
  local status=$?

  conda deactivate || true
  cd || true

  if [[ $status -eq 0 ]]; then
    printf "%s✔ Success%s (exit %d)\n" "$GREEN" "$RESET" "$status"
  else
    printf "%s✖ Failed%s (exit %d)\n" "$RED" "$RESET" "$status"
  fi

  return $status
}

# Quiet runner for the 'lint' pipeline: no command preview, no command output; returns exit code only
run_in_env_quiet() {
  local cmd="$*"
  init_conda || return 1
  conda activate "$ENV_NAME" >/dev/null 2>&1 || return 1
  cd "$PKG_DIR" >/dev/null 2>&1 || { conda deactivate >/dev/null 2>&1 || true; return 1; }
  bash -lc "$cmd" >/dev/null 2>&1
  local status=$?
  conda deactivate >/dev/null 2>&1 || true
  cd >/dev/null 2>&1 || true
  return $status
}

# --- tasks (always from project root) ----------------------------------------
task_poetry_lock() {
  echo "Updating Poetry lock..."
  run_in_env "poetry lock"
}

task_poetry_install() {
  echo "Installing dependencies with Poetry..."
  run_in_env "poetry install"
}

task_precommit() {
  echo "Running pre-commit on all files..."
  run_in_env "poetry run pre-commit run --all-files"
}

task_mypy() {
  echo "Running mypy (idtrack, tests, docs/conf.py)..."
  run_in_env "poetry run mypy idtrack tests docs/conf.py"
}

task_flake8() {
  echo "Running flake8..."
  run_in_env "poetry run flake8 idtrack tests docs/conf.py"
}

task_docs_build() {
  echo "Building HTML docs (no browser)..."
  run_in_env "rm -rf docs/_build && poetry run make -C docs html"
}

task_docs_build_open() {
  echo "Building HTML docs and opening index page..."
  run_in_env "rm -rf docs/_build && poetry run make -C docs html \
    && { \
         if command -v open >/dev/null 2>&1; then open docs/_build/html/index.html; \
         elif command -v xdg-open >/dev/null 2>&1; then xdg-open docs/_build/html/index.html; \
         elif command -v start >/dev/null 2>&1; then start docs/_build/html/index.html; \
         else echo \"Docs built at: $PKG_DIR/docs/_build/html/index.html\"; fi; \
       }"
}

task_pytest_basic() {
  echo "Running tests (pytest basic, typeguard for idtrack)..."
  run_in_env "poetry run pytest --typeguard-packages=idtrack"
}

task_tests_coverage() {
  echo "Running tests with coverage (parallel) and reporting..."
  run_in_env "poetry run coverage run --parallel -m pytest tests \
    && poetry run coverage combine \
    && poetry run coverage report"
}

task_safety() {
  echo "Exporting requirements and running safety check..."
  run_in_env "poetry export -f requirements.txt --output __requirements.txt \
    && poetry run safety check --full-report --file=__requirements.txt || true \
    && rm -f __requirements.txt"
}

# --- lint pipeline (quiet) ---------------------------------------------------
task_lint() {
  echo "Running LINT pipeline (quiet): lock, install, pre-commit, mypy, docs build, pytest, coverage, safety"
  echo "(Skipping: flake; build & open docs)"

  local total=0
  local failed=0

  run_step() {
    local label="$1"; shift
    local cmd="$*"
    total=$((total + 1))
    printf " - %s ... " "$label"
    if run_in_env_quiet "$cmd"; then
      printf "%s✔ Success%s\n" "$GREEN" "$RESET"
    else
      printf "%s✖ Failed%s\n" "$RED" "$RESET"
      failed=$((failed + 1))
    fi
  }

  run_step "Poetry: lock" "poetry lock"
  run_step "Poetry: install" "poetry install"
  run_step "Pre-commit (all files)" "poetry run pre-commit run --all-files"
  run_step "mypy" "poetry run mypy idtrack tests docs/conf.py"
  run_step "Docs: build (no open)" "rm -rf docs/_build && poetry run make -C docs html"
  run_step "Pytest (basic)" "poetry run pytest --typeguard-packages=idtrack"
  run_step "Coverage (tests + report)" "poetry run coverage run --parallel -m pytest tests && poetry run coverage combine && poetry run coverage report"
  # Safety: propagate safety's exit status (fail if vulnerabilities)
  run_step "Safety vulnerability check" "poetry export -f requirements.txt --output __requirements.txt && poetry run safety check --full-report --file=__requirements.txt; s=\$?; rm -f __requirements.txt; exit \$s"

  echo
  if [[ $failed -eq 0 ]]; then
    printf "LINT summary: %s%d/%d passed%s\n" "$GREEN" "$total" "$total" "$RESET"
  else
    printf "LINT summary: %s%d/%d passed%s, %s%d failed%s\n" "$GREEN" "$((total - failed))" "$total" "$RESET" "$RED" "$failed" "$RESET"
  fi
}

# --- menu --------------------------------------------------------------------
print_menu() {
  cat <<EOF

================================================================
IDTrack Dev Menu (env: ${ENV_NAME:-<not selected>})
Project dir: $PKG_DIR
----------------------------------------------------------------
  1) Poetry: lock
  2) Poetry: install
  3) Run pre-commit hooks (all files)
  4) Run mypy
  5) Run flake
  6) Build docs
  7) Build & open docs
  8) Run tests (pytest basic)
  9) Run tests with coverage
 10) Run safety vulnerability check
  l) Lint pipeline (quiet; skip flake & open docs)
  q) Exit
================================================================

EOF
}

# --- main --------------------------------------------------------------------
set -u
init_conda
choose_env

while true; do
  print_menu
  read -rp "Select an option: " choice
  case "$choice" in
    1)  task_poetry_lock ;;
    2)  task_poetry_install ;;
    3)  task_precommit ;;
    4)  task_mypy ;;
    5)  task_flake8 ;;
    6)  task_docs_build ;;
    7)  task_docs_build_open ;;
    8)  task_pytest_basic ;;
    9)  task_tests_coverage ;;
    10) task_safety ;;
    l|L|lint) task_lint ;;
    q|Q) echo "Goodbye!"; exit 0 ;;
    *)  echo "Invalid choice: $choice" ;;
  esac
done
