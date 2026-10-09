#!/usr/bin/env bash
set -eo pipefail
SEARCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SEARCH_DIR/../.."
spec="$1"
shift

source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Eight concurrent jobs: four on GPU 0 and four on GPU 1.
exec python parallel_search.py \
    --config "$SEARCH_DIR/$spec" \
    --workers "${WORKERS:-8}" --gpus "${GPUS:-0,0,0,0,1,1,1,1}" \
    --timeout-hours "${TIMEOUT_HOURS:-48}" "$@"
