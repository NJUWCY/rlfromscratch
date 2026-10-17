#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}"
CONDA_ENV="${CONDA_ENV:-rlzero}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_BASE}/envs/${CONDA_ENV}/bin/python}"
cd "${REPO_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
export RES_OPTIONS="${RES_OPTIONS:-ndots:1}"
export PYTHONUNBUFFERED=1
if [[ $# -eq 0 ]]; then
  set -- --run --gpus 0 --jobs-per-gpu 4 --epochs 1500000 --timeout-hours 48
fi
exec "${PYTHON_BIN}" search/DACSAC/humanoid_wave1.py --mode "${SWANLAB_MODE}" "$@"
