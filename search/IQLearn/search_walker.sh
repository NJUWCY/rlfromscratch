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

# 162 combinations per environment; 2 GPUs with 4 workers each.
# left/right vary independently. Seed 0, 1M interactions per run.
# Example: ENVS=humanoid bash search/IQLearn/iqlearn_param_search.sh
read -r -a env_names <<< "${ENVS:-walker2d humanoid}"
for env_name in "${env_names[@]}"; do
    case "$env_name" in
        ant|halfcheetah|walker2d|humanoid) ;;
        *) printf 'Unsupported environment: %s
' "$env_name" >&2; exit 1 ;;
    esac
done
for env_name in "${env_names[@]}"; do
    python parallel_search.py \
        --config "$SEARCH_DIR/${env_name}_search.json" \
        --workers "${WORKERS:-8}" --gpus "${GPUS:-0,0,0,0,1,1,1,1}" \
        --timeout-hours "${TIMEOUT_HOURS:-48}" "$@"
done
