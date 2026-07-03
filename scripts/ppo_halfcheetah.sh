set -e

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-yyKpLHGppV78RFW0p1PNQ}

ENVS=(
  HalfCheetah-v5
)

for seed in 0 1 2; do
    for env in "${ENVS[@]}"; do 
        python run_ppo.py \
        algorithm=ppo \
        total_epoch=3000 \
        train_log_interval=10 \
        save_interval=1000 \
        test_interval=5 \
        train_action_deterministic=false \
        algorithm.collect_traj=false \
        env.name=${env} \
        algorithm.buffer_name=ReplayBuffer \
        interact_per_epoch=128 \
        algorithm.update_epochs=5 \
        algorithm.minibatch_size=256 \
        algorithm.use_grad_clip=true \
        algorithm.advan_norm=false \
        env.obs_norm=true \
        algorithm.rescale=true \
        algorithm.return_scaling=false \
        seed=${seed} \
        algorithm.initialize=true \
        "$@"
    done 
done