#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export SWANLAB_MODE=online PYTHONUNBUFFERED=1
unset SWANLAB_CORE_SKIP_STORE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TQDM_MININTERVAL=60
exec python search/HyPESAC/humanoid/expert_zero.py "$@"
