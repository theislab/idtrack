#!/bin/bash
#
# Generic Slurm job-array runner for experiment notebooks.
#
# Usage (recommended): submit via `reproducibility/scripts/submit_experiment_notebooks_slurm.sh`.
#
# Required env vars:
#   - MANIFEST: path to a text file listing notebooks (one per line)
#
# Optional env vars:
#   - REPO_ROOT: repo root path (auto-detected if unset)
#   - LOG_DIR: log directory (default: <repo>/reproducibility/experiments/_logs)
#   - CONDA_ENV: conda env name (default: idtrack_dev_env)
#   - IDTRACK_LOCAL_REPO: cache root (default: <repo>/idtrack/docs/_notebooks/idtrack_cache)
#

# -------------------- SLURM defaults (override via sbatch CLI) --------------------
#SBATCH -J idtrack_nb
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --nice=0
#SBATCH -t 22:50:00
#SBATCH --array=0-0
#SBATCH -o slurm_%x_%A_%a.out
#SBATCH -e slurm_%x_%A_%a.err

set -Eeuo pipefail

on_err() {
  local exit_code=$?
  local line_no="${BASH_LINENO[0]:-}"
  local cmd="${BASH_COMMAND:-}"
  echo "[$(date)] ERROR: exit_code=${exit_code} line=${line_no} cmd=${cmd}" >&2
}
trap on_err ERR

echo "[$(date)] BOOT slurm_job_id=${SLURM_JOB_ID:-} slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-}"
echo "[$(date)] BOOT script=${BASH_SOURCE[0]}"

if [[ -z "${REPO_ROOT:-}" ]]; then
  detect_repo_root() {
    local start_dir
    start_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    local candidate="${start_dir}"
    while [[ "${candidate}" != "/" ]]; do
      # Detect either layout:
      # - package-only: <root>/idtrack (python package) + <root>/reproducibility
      # - umbrella:     <root>/idtrack/reproducibility (inside the umbrella checkout)
      if [[ -d "${candidate}/idtrack" && ( -d "${candidate}/reproducibility" || -d "${candidate}/idtrack/reproducibility" ) ]]; then
        echo "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done

    echo "ERROR: Could not detect repo root from ${start_dir}" >&2
    echo "Expected markers: <root>/idtrack and either <root>/reproducibility or <root>/idtrack/reproducibility" >&2
    return 1
  }

  REPO_ROOT="$(detect_repo_root)"
fi

echo "[$(date)] BOOT repo_root=${REPO_ROOT}"

MANIFEST_PATH="${MANIFEST:-}"
if [[ -z "${MANIFEST_PATH}" ]]; then
  echo "ERROR: MANIFEST is required (path to a notebook manifest file)." >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  if [[ -f "${REPO_ROOT}/${MANIFEST_PATH}" ]]; then
    MANIFEST_PATH="${REPO_ROOT}/${MANIFEST_PATH}"
  else
    echo "ERROR: Manifest not found: ${MANIFEST_PATH}" >&2
    exit 1
  fi
fi

echo "[$(date)] BOOT manifest=${MANIFEST_PATH}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "${REPO_ROOT}/idtrack/reproducibility" ]]; then
    LOG_DIR="${REPO_ROOT}/idtrack/reproducibility/experiments/_logs"
  else
    LOG_DIR="${REPO_ROOT}/reproducibility/experiments/_logs"
  fi
fi
NB_LOG_DIR="${NB_LOG_DIR:-$LOG_DIR/nbconvert}"
mkdir -p "${LOG_DIR}" "${NB_LOG_DIR}"

if [[ -z "${EXPERIMENTS_ROOT:-}" ]]; then
  if [[ -d "${REPO_ROOT}/idtrack/reproducibility" ]]; then
    EXPERIMENTS_ROOT="${REPO_ROOT}/idtrack/reproducibility/experiments"
  else
    EXPERIMENTS_ROOT="${REPO_ROOT}/reproducibility/experiments"
  fi
fi

echo "[$(date)] BOOT log_dir=${LOG_DIR}"
echo "[$(date)] BOOT nb_log_dir=${NB_LOG_DIR}"
echo "[$(date)] BOOT experiments_root=${EXPERIMENTS_ROOT}"

cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-idtrack_dev_env}"

# IMPORTANT: Do not source user shell init files by default.
# On some HPC systems, `.bashrc` / `.bash_profile` may contain `exit` or other
# non-interactive-unsafe commands, which would terminate the Slurm job script
# before it even starts the notebook execution.
#
# If you really need site/module initialization from those files, set:
#   SOURCE_SHELL_INIT=1
if [[ "${SOURCE_SHELL_INIT:-0}" == "1" ]]; then
  # Source in a subshell to avoid accidental `exit` killing the job.
  ( source "$HOME/.bashrc" 2>/dev/null || true ) || true
  ( source "$HOME/.bash_profile" 2>/dev/null || true ) || true
fi

# Ensure the experiments helper modules (`experiments_utils.py`, etc) are importable.
export PYTHONPATH="${EXPERIMENTS_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
if [[ -z "${IDTRACK_LOCAL_REPO:-}" ]]; then
  if [[ -d "${REPO_ROOT}/idtrack/docs/_notebooks" ]]; then
    export IDTRACK_LOCAL_REPO="${REPO_ROOT}/idtrack/docs/_notebooks/idtrack_cache"
  else
    export IDTRACK_LOCAL_REPO="${REPO_ROOT}/docs/_notebooks/idtrack_cache"
  fi
else
  export IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO}"
fi

# -------------------- Common experiment input paths (auto-detect) --------------------
# These defaults are derived from the paths used in:
# - `idtrack/docs/_notebooks/05_tutorial_harmonization.ipynb` (HLCA)
# - `idtrack/reproducibility/experiments/experiment_cellranger_idtrack/create_data.ipynb` (gold-standard `.h5ad` outputs)
#
# We only set them if unset AND the directory exists, so this stays portable.
if [[ -z "${HLCA_BASE_PATH:-}" ]]; then
  for candidate in \
    "/lustre/groups/ml01/projects/2023_HLCA_LSikkema/HLCA_reproducibility/data"
  do
    if [[ -d "${candidate}" ]]; then
      export HLCA_BASE_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${GOLD_STANDARD_ANNDATA_DIR:-}" ]]; then
  for candidate in \
    "/home/icb/kemal.inecik/lustre_workspace/idtrack_experiments/anndatas"
  do
    if [[ -d "${candidate}" ]]; then
      export GOLD_STANDARD_ANNDATA_DIR="${candidate}"
      break
    fi
  done
fi
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
echo "[$(date)] BOOT task_id=${TASK_ID}"

# Select the N-th non-empty, non-comment line from the manifest.
set +e
NOTEBOOK_REL="$(grep -vE '^[[:space:]]*($|#)' "${MANIFEST_PATH}" | sed -n "$((TASK_ID + 1))p" | tr -d '\r')"
sel_rc=$?
set -e
if [[ "${sel_rc}" -ne 0 ]]; then
  echo "ERROR: Failed selecting notebook line from manifest (rc=${sel_rc})." >&2
  echo "MANIFEST_PATH=${MANIFEST_PATH}" >&2
  echo "TASK_ID=${TASK_ID}" >&2
  echo "First 30 non-comment entries:" >&2
  grep -vE '^[[:space:]]*($|#)' "${MANIFEST_PATH}" | sed -n '1,30p' >&2 || true
  exit 1
fi

NOTEBOOK_REL="$(echo "${NOTEBOOK_REL}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
echo "[$(date)] BOOT notebook_rel=${NOTEBOOK_REL}"

if [[ -z "${NOTEBOOK_REL}" ]]; then
  echo "ERROR: No notebook entry for SLURM_ARRAY_TASK_ID=${TASK_ID} in ${MANIFEST_PATH}" >&2
  exit 1
fi

resolve_notebook_path() {
  local entry="$1"
  local candidate

  # Absolute paths.
  if [[ "${entry}" == /* && -f "${entry}" ]]; then
    echo "${entry}"
    return 0
  fi

  # Preferred: manifest entries are relative to `${EXPERIMENTS_ROOT}` (layout-agnostic).
  candidate="${EXPERIMENTS_ROOT}/${entry}"
  if [[ -f "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  # Backward-compatible: entries relative to repo root.
  candidate="${REPO_ROOT}/${entry}"
  if [[ -f "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  # Backward-compatible: manifests written for the umbrella layout but executed in package-only layout
  # (e.g., `idtrack/reproducibility/...` vs `reproducibility/...`).
  if [[ "${entry}" == idtrack/* ]]; then
    candidate="${REPO_ROOT}/${entry#idtrack/}"
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

  # Backward-compatible: manifests written for package-only layout but executed in umbrella layout
  # (e.g., `reproducibility/...` vs `idtrack/reproducibility/...`).
  if [[ "${entry}" == reproducibility/* ]]; then
    candidate="${REPO_ROOT}/idtrack/${entry}"
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

  echo "ERROR: Notebook not found for manifest entry: ${entry}" >&2
  echo "Tried:" >&2
  echo "  - ${EXPERIMENTS_ROOT}/${entry}" >&2
  echo "  - ${REPO_ROOT}/${entry}" >&2
  if [[ "${entry}" == idtrack/* ]]; then
    echo "  - ${REPO_ROOT}/${entry#idtrack/}" >&2
  fi
  if [[ "${entry}" == reproducibility/* ]]; then
    echo "  - ${REPO_ROOT}/idtrack/${entry}" >&2
  fi
  return 1
}

set +e
NOTEBOOK_PATH="$(resolve_notebook_path "${NOTEBOOK_REL}")"
resolve_rc=$?
set -e
if [[ "${resolve_rc}" -ne 0 ]]; then
  exit 1
fi

NB_NAME="$(basename "${NOTEBOOK_PATH}" .ipynb)"
RUN_LOG="${NB_LOG_DIR}/${NB_NAME}_${SLURM_JOB_ID}_${TASK_ID}.log"

touch "${RUN_LOG}"

# Mirror all output to the per-notebook run log AND to Slurm stdout/stderr.
# If process substitution is unsupported in this environment, fall back to log-only.
if command -v tee >/dev/null 2>&1; then
  set +e
  exec > >(tee -a "${RUN_LOG}") 2>&1
  tee_rc=$?
  set -e
  if [[ "${tee_rc}" -ne 0 ]]; then
    exec >> "${RUN_LOG}" 2>&1
    echo "[$(date)] WARNING: tee/process-substitution failed; logging only to ${RUN_LOG}" >&2
  fi
else
  exec >> "${RUN_LOG}" 2>&1
  echo "[$(date)] WARNING: tee not found; logging only to ${RUN_LOG}" >&2
fi

echo "[$(date)] START"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Repo: ${REPO_ROOT}"
echo "[$(date)] Manifest: ${MANIFEST_PATH}"
echo "[$(date)] Notebook: ${NOTEBOOK_PATH}"
echo "[$(date)] IDTRACK_LOCAL_REPO: ${IDTRACK_LOCAL_REPO}"
echo "[$(date)] HLCA_BASE_PATH: ${HLCA_BASE_PATH:-}"
echo "[$(date)] GOLD_STANDARD_ANNDATA_DIR: ${GOLD_STANDARD_ANNDATA_DIR:-}"
echo "[$(date)] CONDA_ENV: ${CONDA_ENV}"
echo "[$(date)] SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "[$(date)] SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-}"
echo "[$(date)] SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "[$(date)] SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-}"
echo

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Fallback for common Miniconda installs
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "ERROR: conda not found. Load conda or adjust activation in: ${BASH_SOURCE[0]}" >&2
  exit 1
fi

# Re-assert PYTHONPATH after env activation (some site configs modify it).
export PYTHONPATH="${EXPERIMENTS_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

command -v jupyter >/dev/null 2>&1 || { echo "ERROR: jupyter not found in env ${CONDA_ENV}"; exit 1; }

# Avoid notebook timeouts for long-running cache builds; failure should be a real exception, not a preprocessor timeout.
NBCONVERT_DEBUG="${NBCONVERT_DEBUG:-0}"
DEBUG_FLAG=()
if [[ "${NBCONVERT_DEBUG}" == "1" ]]; then
  DEBUG_FLAG=(--debug)
fi

jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  "${DEBUG_FLAG[@]}" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.iopub_timeout=600 \
  "${NOTEBOOK_PATH}"

echo "[$(date)] DONE"
