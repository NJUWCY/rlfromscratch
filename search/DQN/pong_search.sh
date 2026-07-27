set -e

cd /home/ubuntu/wangchenyang/rlzero/rlfromscratch

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-sac}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-$(tr -d '\r\n' < /home/ubuntu/wangchenyang/RLfromscratch/api.txt)}"
export TQDM_MININTERVAL=60

exec python parallel_search.py \
  --config search/DQN/pong.json \
  --workers 4 \
  --gpus 0,0,0,0 \
  "$@"
