set -e

cd /home/ubuntu/wangchenyang/rlzero/rlfromscratch

export SWANLAB_API_KEY="${SWANLAB_API_KEY:-$(tr -d '\r\n' < /home/ubuntu/wangchenyang/rlzero/rlfromscratch/api.txt)}"

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-sac}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=1

python parallel_search.py \
  --config search/PPO/ppo_search_second.json \
  --workers 8 \
  --gpus 0,0,0,0,1,1,1,1 \
  "$@"

python parallel_search.py \
  --config search/PPO/ppo_search_first.json \
  --workers 8 \
  --gpus 0,0,0,0,1,1,1,1 \
  "$@"
