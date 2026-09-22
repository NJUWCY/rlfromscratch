#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/home/ubuntu/wangchenyang/rlzero/rlfromscratch
cd "${REPO_DIR}"

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-$(tr -d '\r\n' < "${REPO_DIR}/api.txt")}" 
export TQDM_MININTERVAL=60
export RES_OPTIONS="${RES_OPTIONS:-ndots:1}"
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"

# Broad first-stage random search. The full Cartesian grid has 324 settings;
# sample a reproducible subset because full Ant runs are expensive.
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,0,0,0,1,1,1,1}
LIMIT=${LIMIT:-64}
SEARCH_SEED=${SEARCH_SEED:-20260916}
TIMEOUT_HOURS=${TIMEOUT_HOURS:-12}

exec python parallel_search.py \
  --config search/AIRLPPO/ant_airlppo_search.json \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --shuffle \
  --limit "${LIMIT}" \
  --seed "${SEARCH_SEED}" \
  --timeout-hours "${TIMEOUT_HOURS}" \
  "$@"
