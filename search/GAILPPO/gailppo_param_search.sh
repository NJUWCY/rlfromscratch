set -e

REPO_DIR=/home/ubuntu/wangchenyang/rlzero/rlfromscratch

cd "${REPO_DIR}"

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-$(tr -d '\r\n' < "${REPO_DIR}/api.txt")}"
export TQDM_MININTERVAL=60

# The cluster resolver uses ndots:5, so every lookup of api.swanlab.cn first tries
# three in-cluster search domains. With 8 concurrent jobs that amplifies load on
# CoreDNS and causes the intermittent "Temporary failure in name resolution".
export RES_OPTIONS="${RES_OPTIONS:-ndots:1}"
# online | offline | disabled. Use offline on unreliable networks and upload later
# with: swanlab sync <run-dir>
export SWANLAB_MODE="${SWANLAB_MODE:-online}"

# 8 concurrent jobs spread round-robin over 2 GPUs (4 jobs per GPU).
# A run takes ~7.5h, so the timeout only fires on a job stuck in shutdown.
exec python parallel_search.py \
  --config search/GAILPPO/gailppo_search.json \
  --workers 8 \
  --gpus 0,0,0,0,1,1,1,1 \
  --timeout-hours "${TIMEOUT_HOURS:-10}" \
  "$@"
