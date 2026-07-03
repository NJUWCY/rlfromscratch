set -e

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-yyKpLHGppV78RFW0p1PNQ}

GAMES=(
  BreakoutNoFrameskip-v4
  SeaquestNoFrameskip-v4
  QbertNoFrameskip-v4
  SpaceInvadersNoFrameskip-v4
  BeamRiderNoFrameskip-v4
  PongNoFrameskip-v4
  FreewayNoFrameskip-v4
)

for game in "${GAMES[@]}"; do
    python run_ppo_atari.py \
    algorithm=ppo \
    env=atari \
    total_epoch=10000 \
    train_log_interval=10 \
    save_interval=1000 \
    test_interval=50 \
    train_action_deterministic=false \
    algorithm.collect_traj=false \
    env.name=${game} \
    algorithm.buffer_name=ReplayBuffer \
    interact_per_epoch=128 \
    algorithm.update_epochs=4 \
    algorithm.minibatch_size=256 \
    algorithm.use_grad_clip=true \
    algorithm.advan_norm=true \
    algorithm.rescale=false \
    algorithm.return_scaling=false \
    algorithm.actor_lr=2.5e-4 \
    algorithm.critic_lr=2.5e-4 \
    algorithm.entropy_coef=0.01 \
    algorithm.eps_clip=0.1 \
    algorithm.value_clip=0.1 \
    algorithm.common_head=true \
    env.scale=true \
    env.obs_norm=true \
    "$@"

done