#!/bin/bash
#
# Generic Slurm job-array runner for experiment notebooks.
#
# Usage (recommended): submit via `submit_experiment_notebooks_slurm.sh`.
#
# Required env vars:
#   - MANIFEST: path to a text file listing notebooks (one per line)
#
# Optional env vars:
#   - REPO_ROOT: repo root path (auto-detected if unset)
#   - LOG_DIR: log directory (default: <repo>/idtrack/reproducibility/experiments/_logs)
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
#SBATCH -o idtrack/reproducibility/experiments/_logs/slurm_%x_%A_%a.out
#SBATCH -e idtrack/reproducibility/experiments/_logs/slurm_%x_%A_%a.err

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

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

LOG_DIR="${LOG_DIR:-$REPO_ROOT/idtrack/reproducibility/experiments/_logs}"
NB_LOG_DIR="${NB_LOG_DIR:-$LOG_DIR/nbconvert}"
mkdir -p "${LOG_DIR}" "${NB_LOG_DIR}"

CONDA_ENV="${CONDA_ENV:-idtrack_dev_env}"

source "$HOME/.bashrc" 2>/dev/null || true
source "$HOME/.bash_profile" 2>/dev/null || true

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

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO:-$REPO_ROOT/idtrack/docs/_notebooks/idtrack_cache}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Select the N-th non-empty, non-comment line from the manifest.
NOTEBOOK_REL="$(grep -vE '^[[:space:]]*($|#)' "${MANIFEST_PATH}" | sed -n "$((TASK_ID + 1))p" | tr -d '\r')"
NOTEBOOK_REL="$(echo "${NOTEBOOK_REL}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"

if [[ -z "${NOTEBOOK_REL}" ]]; then
  echo "ERROR: No notebook entry for SLURM_ARRAY_TASK_ID=${TASK_ID} in ${MANIFEST_PATH}" >&2
  exit 1
fi

NOTEBOOK_PATH="${NOTEBOOK_REL}"
if [[ "${NOTEBOOK_PATH}" != /* ]]; then
  NOTEBOOK_PATH="${REPO_ROOT}/${NOTEBOOK_REL}"
fi

if [[ ! -f "${NOTEBOOK_PATH}" ]]; then
  echo "ERROR: Notebook not found: ${NOTEBOOK_PATH}" >&2
  exit 1
fi

NB_NAME="$(basename "${NOTEBOOK_PATH}" .ipynb)"
RUN_LOG="${NB_LOG_DIR}/${NB_NAME}_${SLURM_JOB_ID}_${TASK_ID}.log"

{
  echo "[$(date)] START"
  echo "[$(date)] Host: $(hostname)"
  echo "[$(date)] Repo: ${REPO_ROOT}"
  echo "[$(date)] Manifest: ${MANIFEST_PATH}"
  echo "[$(date)] Notebook: ${NOTEBOOK_PATH}"
  echo "[$(date)] IDTRACK_LOCAL_REPO: ${IDTRACK_LOCAL_REPO}"
  echo "[$(date)] CONDA_ENV: ${CONDA_ENV}"
  echo "[$(date)] SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
  echo "[$(date)] SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-}"
  echo "[$(date)] SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
  echo "[$(date)] SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-}"
  echo
} | tee -a "${RUN_LOG}"

command -v jupyter >/dev/null 2>&1 || { echo "ERROR: jupyter not found in env ${CONDA_ENV}" | tee -a "${RUN_LOG}"; exit 1; }

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
  "${NOTEBOOK_PATH}" >> "${RUN_LOG}" 2>&1

echo "[$(date)] DONE" | tee -a "${RUN_LOG}"
