#!/bin/bash
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
  echo "===== Running DQN on ${game} ====="
  python run_dqn.py \
    algorithm=dqn \
    train_action_deterministic=true \
    env=atari \
    env.name=${game} \
    env.num_training_envs=1 \
    save_interval=100000 \
    test_interval=10000 \
    train_log_interval=5000 \
    total_epoch=2500000 \
    start_train_step=20000 \
    algorithm.buffer_size=1000000 \
    algorithm.huber_loss=true \
    algorithm.end_epsilon=0.01 \
    algorithm.target_update_interval=2000 \
    algorithm.target_update_tau=1 \
    algorithm.doubledqn=true \
    algorithm.buffer_name="PrioritizedReplayBuffer" \
    algorithm.weight_batch_norm=true \
    algorithm.learning_rate=5e-5 \
    algorithm.epsilon_timestep=250000 \
    "$@"
  echo "===== Finished ${game} ====="
done
