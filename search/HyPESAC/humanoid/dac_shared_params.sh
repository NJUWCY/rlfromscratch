#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export SWANLAB_MODE=offline PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TQDM_MININTERVAL=60
exec python search/HyPESAC/humanoid/dac_shared_params.py "$@"
