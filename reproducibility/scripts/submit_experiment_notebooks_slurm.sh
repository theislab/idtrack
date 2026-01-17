#!/bin/bash
#
# Submit all manuscript-facing experiment notebooks to Slurm as job arrays.
#
# Usage:
#   ./idtrack/reproducibility/scripts/submit_experiment_notebooks_slurm.sh
#
# Optional env vars:
#   - CONDA_ENV: conda env name (default: idtrack_dev_env)
#   - IDTRACK_LOCAL_REPO: cache root (default: <repo>/idtrack/docs/_notebooks/idtrack_cache)
#   - MAX_PARALLEL: maximum concurrent array tasks per stage (default: 3)
#   - STAGE0_MANIFEST: override stage0 manifest path
#   - STAGE1_MANIFEST: override stage1 manifest path
#   - STAGE2_MANIFEST: override stage2 manifest path
#
# Notes:
#   - Stage 1 depends on Stage 0 (cache build dependency).
#   - Stage 2 depends on Stage 1 (summary/dashboard notebooks).
#   - Logs are written under: idtrack/reproducibility/experiments/_logs/
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-idtrack_dev_env}"
IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO:-$REPO_ROOT/idtrack/docs/_notebooks/idtrack_cache}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

LOG_DIR="${REPO_ROOT}/idtrack/reproducibility/experiments/_logs"
mkdir -p "${LOG_DIR}"

SBATCH_SCRIPT="idtrack/reproducibility/scripts/run_experiment_notebooks_array.sbatch"

STAGE0_MANIFEST="${STAGE0_MANIFEST:-idtrack/reproducibility/experiments/notebooks_manifest_stage0.txt}"
STAGE1_MANIFEST="${STAGE1_MANIFEST:-idtrack/reproducibility/experiments/notebooks_manifest_stage1.txt}"
STAGE2_MANIFEST="${STAGE2_MANIFEST:-idtrack/reproducibility/experiments/notebooks_manifest_stage2.txt}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: Missing sbatch runner: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

count_notebooks() {
  local manifest="$1"
  if [[ ! -f "${manifest}" ]]; then
    echo "0"
    return
  fi
  grep -vE '^[[:space:]]*($|#)' "${manifest}" | wc -l | tr -d ' '
}

N0="$(count_notebooks "${STAGE0_MANIFEST}")"
N1="$(count_notebooks "${STAGE1_MANIFEST}")"
N2="$(count_notebooks "${STAGE2_MANIFEST}")"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "IDTRACK_LOCAL_REPO: ${IDTRACK_LOCAL_REPO}"
echo "Log dir: ${LOG_DIR}"
echo "Stage0 manifest: ${STAGE0_MANIFEST} (${N0} notebooks)"
echo "Stage1 manifest: ${STAGE1_MANIFEST} (${N1} notebooks)"
echo "Stage2 manifest: ${STAGE2_MANIFEST} (${N2} notebooks)"
echo

JOB0=""
if [[ "${N0}" -gt 0 ]]; then
  JOB0="$(
    sbatch --parsable \
      --array=0-$((N0 - 1))%${MAX_PARALLEL} \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",MANIFEST="${STAGE0_MANIFEST}",LOG_DIR="${LOG_DIR}",CONDA_ENV="${CONDA_ENV}",IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO}" \
      "${SBATCH_SCRIPT}"
  )"
  echo "Submitted stage0: ${JOB0}"
fi

JOB1=""
if [[ "${N1}" -gt 0 ]]; then
  DEP=()
  if [[ -n "${JOB0}" ]]; then
    DEP=(--dependency="afterok:${JOB0}")
  fi

  JOB1="$(
    sbatch --parsable "${DEP[@]}" \
      --array=0-$((N1 - 1))%${MAX_PARALLEL} \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",MANIFEST="${STAGE1_MANIFEST}",LOG_DIR="${LOG_DIR}",CONDA_ENV="${CONDA_ENV}",IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO}" \
      "${SBATCH_SCRIPT}"
  )"
  echo "Submitted stage1: ${JOB1}"
fi

if [[ "${N2}" -gt 0 ]]; then
  DEP2=()
  if [[ -n "${JOB1}" ]]; then
    DEP2=(--dependency="afterok:${JOB1}")
  elif [[ -n "${JOB0}" ]]; then
    DEP2=(--dependency="afterok:${JOB0}")
  fi

  JOB2="$(
    sbatch --parsable "${DEP2[@]}" \
      --array=0-$((N2 - 1))%${MAX_PARALLEL} \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",MANIFEST="${STAGE2_MANIFEST}",LOG_DIR="${LOG_DIR}",CONDA_ENV="${CONDA_ENV}",IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO}" \
      "${SBATCH_SCRIPT}"
  )"
  echo "Submitted stage2: ${JOB2}"
fi

echo
echo "Tip: tail logs under ${LOG_DIR}/nbconvert/"
