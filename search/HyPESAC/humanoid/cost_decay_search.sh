#!/usr/bin/env bash
set -euo pipefail
export WORKERS=8
export GPUS=0,0,0,0,1,1,1,1
export SWANLAB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" cost_decay_search.json "$@"
