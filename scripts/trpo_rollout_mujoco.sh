#!/usr/bin/env bash
set -e

cd /home/ubuntu/wangchenyang/RLfromscratch

source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate /home/ubuntu/wangchenyang/anaconda/envs/sac

export PYTHONUNBUFFERED=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-$(tr -d '\r\n' < /home/ubuntu/wangchenyang/RLfromscratch/api.txt)}"

seeds=(0 1 2)
env_names=(
  "Hopper-v5"
  "Walker2d-v5"
  "HalfCheetah-v5"
  "Ant-v5"
  "Humanoid-v5"
)

for seed in "${seeds[@]}"; do
  for env_name in "${env_names[@]}"; do
    python run_trpo.py \
      algorithm=trpo \
      total_epoch=3000 \
      train_log_interval=10 \
      save_interval=1000 \
      test_interval=5 \
      train_action_deterministic=false \
      algorithm.rescale=true \
      algorithm.collect_traj=false \
      env=mujoco \
      env.name="${env_name}" \
      env.num_training_envs=8 \
      seed="${seed}" \
      use_swanlab=true \
      algorithm.buffer_name=ReplayBuffer \
      interact_per_epoch=128 \
      algorithm.critic_update_steps=5 \
      algorithm.linesearch_coeffient=0.8 \
      algorithm.advan_norm=false \
      algorithm.return_scaling=true \
      algorithm.initialize=true \
      "$@"
  done
done
