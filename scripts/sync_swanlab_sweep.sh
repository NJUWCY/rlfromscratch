#!/usr/bin/env bash
set -euo pipefail

SWEEP_DIR="${1:?Usage: sync_swanlab_sweep.sh <sweep_dir> [workers] [project]}"
WORKERS="${2:-4}"
PROJECT="${3:-}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}"
CONDA_ENV="${CONDA_ENV:-rlzero}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY="${SWANLAB_API_KEY:-$(tr -d '\r\n' < "${REPO_DIR}/api.txt")}"
export RES_OPTIONS="${RES_OPTIONS:-ndots:1}"

LOG_FILE="${SWEEP_DIR}/swanlab_sync.log"
mapfile -t RUN_DIRS < <(find "${SWEEP_DIR}" -type d -name 'run-*' | sort)
TOTAL="${#RUN_DIRS[@]}"

echo "[$(date -Is)] Start syncing ${TOTAL} runs from ${SWEEP_DIR} (workers=${WORKERS}${PROJECT:+, project=${PROJECT}})" | tee "${LOG_FILE}"

sync_one() {
  local idx="$1" run_dir="$2"
  local extra=()
  if [[ -n "${PROJECT}" ]]; then
    extra+=(-w ChenyangWang -p "${PROJECT}")
  fi
  if swanlab sync "${extra[@]}" "${run_dir}" >/tmp/swanlab_sync_"$$"_"${idx}".out 2>&1; then
    echo "[$(date -Is)] [${idx}/${TOTAL}] OK  ${run_dir}"
  else
    echo "[$(date -Is)] [${idx}/${TOTAL}] FAIL(${run_dir})"
    sed 's/^/  /' "/tmp/swanlab_sync_$$_${idx}.out"
    return 1
  fi
}

export -f sync_one
export TOTAL PROJECT LOG_FILE

ok=0
fail=0
idx=0
# Simple parallel pool via background jobs.
running=0
pids=()
statuses=()
idx=0
for run_dir in "${RUN_DIRS[@]}"; do
  idx=$((idx + 1))
  (
    if sync_one "${idx}" "${run_dir}"; then
      exit 0
    else
      exit 1
    fi
  ) >> "${LOG_FILE}" 2>&1 &
  pids+=("$!")
  running=$((running + 1))
  if (( running >= WORKERS )); then
    wait "${pids[0]}" && ok=$((ok + 1)) || fail=$((fail + 1))
    pids=("${pids[@]:1}")
    running=$((running - 1))
  fi
done
for pid in "${pids[@]}"; do
  wait "${pid}" && ok=$((ok + 1)) || fail=$((fail + 1))
done

echo "[$(date -Is)] Done. OK=${ok} FAIL=${fail} TOTAL=${TOTAL}" | tee -a "${LOG_FILE}"
exit $(( fail > 0 ? 1 : 0 ))
