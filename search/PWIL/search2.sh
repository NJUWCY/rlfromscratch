#!/usr/bin/env bash
set -eo pipefail
SEARCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SEARCH_DIR/../.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# 81 runs per environment; 8 workers on 2 GPUs (4 workers per GPU).
# Set ENVS=humanoid to run only Humanoid.
read -r -a env_names <<< "${ENVS:-halfcheetah}"
for env_name in "${env_names[@]}"; do
    python parallel_search.py \
        --config "$SEARCH_DIR/${env_name}_search.json" \
        --workers "${WORKERS:-8}" --gpus "${GPUS:-0,0,0,0,1,1,1,1}" \
        --timeout-hours "${TIMEOUT_HOURS:-48}" "$@"
done
