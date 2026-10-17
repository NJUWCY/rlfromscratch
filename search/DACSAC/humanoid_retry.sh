#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export SWANLAB_MODE=offline PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TQDM_MININTERVAL=60
# Eight candidates: four per GPU. Incomplete checkpoints cannot resume training.
if [[ " $* " != *" --dry-run "* ]]; then
    python -c 'import torch; from pathlib import Path; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Two GPUs required"; assert Path("outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5").is_file(), "Expert data missing"'
fi
exec python parallel_search.py \
    --config search/DACSAC/humanoid_retry.json \
    --workers 8 --gpus 0,1,0,1,0,1,0,1 --timeout-hours 48 "$@"
