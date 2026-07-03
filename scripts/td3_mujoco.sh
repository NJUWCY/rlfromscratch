set -e

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-yyKpLHGppV78RFW0p1PNQ}

ENVS=(
   Ant-v5
   Hopper-v5
   HalfCheetah-v5
   Humanoid-v5
   Walker2d-v5
)

for seed in 0 1 2; do 
    for env in "${ENVS[@]}"; do
        python run_td3.py \
        algorithm=td3 \
        train_action_deterministic=false \
        env=mujoco \
        env.name=${env} \
        env.num_training_envs=1 \
        save_interval=100000 \
        test_interval=10000 \
        train_log_interval=10000 \
        total_epoch=1500000 \
        algorithm.buffer_size=1000000 \
        interact_per_epoch=1 \
        start_train_step=25000 \
        "$@"
    done

done 