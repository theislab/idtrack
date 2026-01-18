#!/bin/bash
#
# Submit all manuscript-facing experiment notebooks to Slurm as job arrays.
#
# Usage:
#   ./reproducibility/scripts/submit_experiment_notebooks_slurm.sh
#   # (If your checkout is an umbrella repo, prefix with `idtrack/`.)
#
# Optional env vars:
#   - CONDA_ENV: conda env name (default: idtrack_dev_env)
#   - IDTRACK_LOCAL_REPO: cache root (default: <repo>/idtrack/docs/_notebooks/idtrack_cache)
#   - STAGE0_MANIFEST: override stage0 manifest path
#   - STAGE1_MANIFEST: override stage1 manifest path
#   - STAGE2_MANIFEST: override stage2 manifest path
#   - SUBMIT_STAGE0: 1/0 (default: 1)
#   - SUBMIT_STAGE1: 1/0 (default: 1)
#   - SUBMIT_STAGE2: 1/0 (default: 1)
#   - DEPENDENCY_MODE: afterok|afterany (default: afterok)
#   - STAGE2_DEPENDENCY_MODE: afterok|afterany (default: afterany)
#   - AUTO_SKIP_MISSING_INPUTS: 1/0 (default: 0)
#       - If enabled, notebooks that require missing input env vars are removed from the manifests at submit time.
#
# Notes:
#   - Stage 1 depends on Stage 0 (cache build dependency).
#   - Stage 2 depends on Stage 1 (summary/dashboard notebooks).
#   - Logs are written under: idtrack/reproducibility/experiments/_logs/
#

set -euo pipefail

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
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-idtrack_dev_env}"
SUBMIT_STAGE0="${SUBMIT_STAGE0:-1}"
SUBMIT_STAGE1="${SUBMIT_STAGE1:-1}"
SUBMIT_STAGE2="${SUBMIT_STAGE2:-1}"
DEPENDENCY_MODE="${DEPENDENCY_MODE:-afterok}"
STAGE2_DEPENDENCY_MODE="${STAGE2_DEPENDENCY_MODE:-afterany}"
AUTO_SKIP_MISSING_INPUTS="${AUTO_SKIP_MISSING_INPUTS:-0}"

case "${DEPENDENCY_MODE}" in
  afterok|afterany) ;;
  *)
    echo "ERROR: DEPENDENCY_MODE must be 'afterok' or 'afterany' (got: ${DEPENDENCY_MODE})" >&2
    exit 1
    ;;
esac

case "${STAGE2_DEPENDENCY_MODE}" in
  afterok|afterany) ;;
  *)
    echo "ERROR: STAGE2_DEPENDENCY_MODE must be 'afterok' or 'afterany' (got: ${STAGE2_DEPENDENCY_MODE})" >&2
    exit 1
    ;;
esac

# Resolve layout-specific paths.
REPRO_ROOT=""
if [[ -d "${REPO_ROOT}/idtrack/reproducibility" ]]; then
  # Umbrella layout
  REPRO_ROOT="${REPO_ROOT}/idtrack/reproducibility"
  IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO:-$REPO_ROOT/idtrack/docs/_notebooks/idtrack_cache}"
else
  # Package-only layout
  REPRO_ROOT="${REPO_ROOT}/reproducibility"
  IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO:-$REPO_ROOT/docs/_notebooks/idtrack_cache}"
fi

LOG_DIR="${REPRO_ROOT}/experiments/_logs"
mkdir -p "${LOG_DIR}"

SBATCH_SCRIPT="${REPRO_ROOT}/scripts/run_experiment_notebooks_array.sbatch"

# Auto-detect common experiment inputs based on notebook-configured defaults (only if dirs exist).
if [[ -z "${HLCA_BASE_PATH:-}" ]]; then
  if [[ -d "/lustre/groups/ml01/projects/2023_HLCA_LSikkema/HLCA_reproducibility/data" ]]; then
    export HLCA_BASE_PATH="/lustre/groups/ml01/projects/2023_HLCA_LSikkema/HLCA_reproducibility/data"
  fi
fi

if [[ -z "${GOLD_STANDARD_ANNDATA_DIR:-}" ]]; then
  if [[ -d "/home/icb/kemal.inecik/lustre_workspace/idtrack_experiments/anndatas" ]]; then
    export GOLD_STANDARD_ANNDATA_DIR="/home/icb/kemal.inecik/lustre_workspace/idtrack_experiments/anndatas"
  fi
fi

default_manifest() {
  local rel="$1"
  if [[ -f "${REPRO_ROOT}/experiments/${rel}" ]]; then
    echo "${REPRO_ROOT}/experiments/${rel}"
    return 0
  fi
  # Fallback for older umbrella manifests that may still include the `idtrack/` prefix.
  if [[ -f "${REPO_ROOT}/${rel}" ]]; then
    echo "${REPO_ROOT}/${rel}"
    return 0
  fi
  echo "${REPRO_ROOT}/experiments/${rel}"
}

STAGE0_MANIFEST="${STAGE0_MANIFEST:-$(default_manifest notebooks_manifest_stage0.txt)}"
STAGE1_MANIFEST="${STAGE1_MANIFEST:-$(default_manifest notebooks_manifest_stage1.txt)}"
STAGE2_MANIFEST="${STAGE2_MANIFEST:-$(default_manifest notebooks_manifest_stage2.txt)}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: Missing sbatch runner: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

maybe_filter_manifest() {
  local manifest="$1"
  local label="$2"

  if [[ "${AUTO_SKIP_MISSING_INPUTS}" != "1" ]]; then
    echo "${manifest}"
    return 0
  fi

  local skip_hlca="0"
  local skip_gold="0"
  if [[ -z "${HLCA_BASE_PATH:-}" ]]; then
    skip_hlca="1"
  fi
  if [[ -z "${GOLD_STANDARD_ANNDATA_DIR:-}" ]]; then
    skip_gold="1"
  fi

  if [[ "${skip_hlca}" == "0" && "${skip_gold}" == "0" ]]; then
    echo "${manifest}"
    return 0
  fi

  local filtered
  filtered="$(mktemp "${LOG_DIR}/${label}_manifest_filtered.XXXXXX.txt")"

  awk -v skip_hlca="${skip_hlca}" -v skip_gold="${skip_gold}" '
    /^[[:space:]]*($|#)/ { print; next }
    (skip_hlca == "1" && $0 ~ /experiment_hlca\//) { next }
    (skip_gold == "1" && $0 ~ /experiment_cellranger_idtrack\//) { next }
    { print }
  ' "${manifest}" > "${filtered}"

  echo "${filtered}"
}

count_notebooks() {
  local manifest="$1"
  if [[ ! -f "${manifest}" ]]; then
    echo "0"
    return
  fi
  grep -vE '^[[:space:]]*($|#)' "${manifest}" | wc -l | tr -d ' '
}

manifest_entries() {
  local manifest="$1"
  grep -vE '^[[:space:]]*($|#)' "${manifest}" | sed 's/\r$//'
}

find_index_by_suffix() {
  local manifest="$1"
  local suffix="$2"
  local i=0
  local line
  while IFS= read -r line; do
    line="$(echo "${line}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    if [[ -z "${line}" ]]; then
      continue
    fi
    if [[ "${line}" == "${suffix}" || "${line}" == *"${suffix}" ]]; then
      echo "${i}"
      return 0
    fi
    i=$((i + 1))
  done < <(manifest_entries "${manifest}")
  return 1
}

STAGE0_MANIFEST="$(maybe_filter_manifest "${STAGE0_MANIFEST}" "stage0")"
STAGE1_MANIFEST="$(maybe_filter_manifest "${STAGE1_MANIFEST}" "stage1")"
STAGE2_MANIFEST="$(maybe_filter_manifest "${STAGE2_MANIFEST}" "stage2")"

N0="$(count_notebooks "${STAGE0_MANIFEST}")"
N1="$(count_notebooks "${STAGE1_MANIFEST}")"
N2="$(count_notebooks "${STAGE2_MANIFEST}")"

echo "Repo root: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "IDTRACK_LOCAL_REPO: ${IDTRACK_LOCAL_REPO}"
echo "HLCA_BASE_PATH: ${HLCA_BASE_PATH:-}"
echo "GOLD_STANDARD_ANNDATA_DIR: ${GOLD_STANDARD_ANNDATA_DIR:-}"
echo "Log dir: ${LOG_DIR}"
echo "Stage0 manifest: ${STAGE0_MANIFEST} (${N0} notebooks)"
echo "Stage1 manifest: ${STAGE1_MANIFEST} (${N1} notebooks)"
echo "Stage2 manifest: ${STAGE2_MANIFEST} (${N2} notebooks)"
echo

# Preflight warnings for common required inputs (not fatal).
manifest_has() {
  local manifest="$1"
  local pattern="$2"
  [[ -f "${manifest}" ]] || return 1
  grep -vE '^[[:space:]]*($|#)' "${manifest}" | grep -qE "${pattern}"
}

warn_if_unset() {
  local var_name="$1"
  local hint="$2"
  if [[ -z "${!var_name:-}" ]]; then
    echo "WARNING: ${var_name} is not set. ${hint}" >&2
  fi
}

if manifest_has "${STAGE0_MANIFEST}" 'experiment_hlca/' || manifest_has "${STAGE1_MANIFEST}" 'experiment_hlca/'; then
  warn_if_unset "HLCA_BASE_PATH" "HLCA notebooks may fail early without input data."
fi
if manifest_has "${STAGE0_MANIFEST}" 'experiment_cellranger_idtrack/' || manifest_has "${STAGE1_MANIFEST}" 'experiment_cellranger_idtrack/'; then
  warn_if_unset "GOLD_STANDARD_ANNDATA_DIR" 'Gold-standard analysis notebooks may fail if `.h5ad` inputs are not available.'
fi

# Force Slurm stdout/stderr into the same log dir (robust against varying workdirs).
SBATCH_OUT="${LOG_DIR}/slurm_%x_%A_%a.out"
SBATCH_ERR="${LOG_DIR}/slurm_%x_%A_%a.err"

JOB0=""
if [[ "${N0}" -gt 0 && "${SUBMIT_STAGE0}" == "1" ]]; then
  JOB0="$(
    sbatch --parsable \
      --array=0-$((N0 - 1)) \
      --output="${SBATCH_OUT}" \
      --error="${SBATCH_ERR}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",MANIFEST="${STAGE0_MANIFEST}",LOG_DIR="${LOG_DIR}",CONDA_ENV="${CONDA_ENV}",IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO}" \
      "${SBATCH_SCRIPT}"
  )"
  echo "Submitted stage0: ${JOB0}"
fi

JOB1=""
if [[ "${N1}" -gt 0 && "${SUBMIT_STAGE1}" == "1" ]]; then
  # Submit stage1 as N single-task jobs so we can attach precise dependencies
  # (avoids blocking the entire stage1 if any unrelated stage0 notebook fails).
  #
  # Known cache-build dependencies:
  # - time_travel_matrix/* depends on stage0 time_travel_matrix/00_build_time_travel_matrix_cache.ipynb
  # - time_travel_vs_external_mappers/* depends on stage0 time_travel_vs_external_mappers/00_build_time_travel_vs_external_mappers_cache.ipynb

  declare -a STAGE1_JOBIDS=()

  TIME_TRAVEL_BUILD_IDX=""
  TIME_TRAVEL_VS_BUILD_IDX=""
  if [[ -n "${JOB0}" ]]; then
    TIME_TRAVEL_BUILD_IDX="$(find_index_by_suffix "${STAGE0_MANIFEST}" "experiment_time_travel_matrix/00_build_time_travel_matrix_cache.ipynb" || true)"
    TIME_TRAVEL_VS_BUILD_IDX="$(find_index_by_suffix "${STAGE0_MANIFEST}" "experiment_time_travel_vs_external_mappers/00_build_time_travel_vs_external_mappers_cache.ipynb" || true)"
  fi

  for i in $(seq 0 $((N1 - 1))); do
    nb_rel="$(manifest_entries "${STAGE1_MANIFEST}" | sed -n "$((i + 1))p")"
    nb_rel="$(echo "${nb_rel}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"

    dep_spec=""
    if [[ -n "${JOB0}" ]]; then
      if [[ "${nb_rel}" == *"experiment_time_travel_matrix/"* && -n "${TIME_TRAVEL_BUILD_IDX}" ]]; then
        dep_spec="${JOB0}_${TIME_TRAVEL_BUILD_IDX}"
      elif [[ "${nb_rel}" == *"experiment_time_travel_vs_external_mappers/"* && -n "${TIME_TRAVEL_VS_BUILD_IDX}" ]]; then
        dep_spec="${JOB0}_${TIME_TRAVEL_VS_BUILD_IDX}"
      fi
    fi

    dep_args=()
    if [[ -n "${dep_spec}" ]]; then
      dep_args=(--dependency="${DEPENDENCY_MODE}:${dep_spec}")
    fi

    jobid="$(
      sbatch --parsable "${dep_args[@]}" \
        --array="${i}-${i}" \
        --output="${SBATCH_OUT}" \
        --error="${SBATCH_ERR}" \
        --export=ALL,REPO_ROOT="${REPO_ROOT}",MANIFEST="${STAGE1_MANIFEST}",LOG_DIR="${LOG_DIR}",CONDA_ENV="${CONDA_ENV}",IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO}" \
        "${SBATCH_SCRIPT}"
    )"

    STAGE1_JOBIDS+=("${jobid}")
    echo "Submitted stage1[$i]: ${jobid}  (${nb_rel})"
  done

  JOB1="${STAGE1_JOBIDS[*]}"
fi

if [[ "${N2}" -gt 0 && "${SUBMIT_STAGE2}" == "1" ]]; then
  # Stage2 is a summary/dashboard step; it should generally run once upstream jobs are done,
  # even if some optional notebooks failed. By default we use `afterany` for stage2.
  DEP2=()
  dep_ids=()
  if [[ -n "${JOB0}" ]]; then
    dep_ids+=("${JOB0}")
  fi
  if [[ -n "${JOB1}" ]]; then
    # JOB1 is a space-separated list of job ids (stage1 submitted as single-task jobs).
    for jid in ${JOB1}; do
      dep_ids+=("${jid}")
    done
  fi
  if [[ "${#dep_ids[@]}" -gt 0 ]]; then
    DEP2=(--dependency="${STAGE2_DEPENDENCY_MODE}:$(IFS=:; echo "${dep_ids[*]}")")
  fi

  JOB2="$(
    sbatch --parsable "${DEP2[@]}" \
      --array=0-$((N2 - 1)) \
      --output="${SBATCH_OUT}" \
      --error="${SBATCH_ERR}" \
      --export=ALL,REPO_ROOT="${REPO_ROOT}",MANIFEST="${STAGE2_MANIFEST}",LOG_DIR="${LOG_DIR}",CONDA_ENV="${CONDA_ENV}",IDTRACK_LOCAL_REPO="${IDTRACK_LOCAL_REPO}" \
      "${SBATCH_SCRIPT}"
  )"
  echo "Submitted stage2: ${JOB2}"
fi

echo
echo "Tip: tail logs under ${LOG_DIR}/nbconvert/"
echo "Note: stage1 submits per-notebook jobs with targeted dependencies on cache-build notebooks (avoids global DependencyNeverSatisfied)."
echo "Note: stage2 waits for stage0+stage1 by default via STAGE2_DEPENDENCY_MODE=afterany."

echo
echo "Debug helpers:"
if [[ -n "${JOB0}" ]]; then
  echo "- Stage0 status: sacct -j ${JOB0} --format=JobID,State,ExitCode,Elapsed,NodeList%30,Reason%40"
  echo "- Stage0 logs:   ls -1 ${LOG_DIR}/slurm_idtrack_nb_${JOB0}_*.err | head"
fi
if [[ -n "${JOB1}" ]]; then
  echo "- Stage1 jobids: ${JOB1}"
fi
echo "- Notebook logs: ls -1 ${LOG_DIR}/nbconvert/ | tail"
