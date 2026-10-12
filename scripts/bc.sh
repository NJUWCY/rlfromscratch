#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

env_names=(Ant-v5 HalfCheetah-v5 Hopper-v5 Walker2d-v5 Humanoid-v5)
for arg in "$@"; do
    if [[ "$arg" == env.name=* ]]; then env_names=("${arg#env.name=}"); fi
done

for env_name in "${env_names[@]}"; do
    python run_bc.py \
        algorithm=bc env=mujoco env.name="${env_name}" \
        env.num_training_envs=1 env.num_testing_envs=1 \
        env.obs_norm=false env.absorbing=false \
        algorithm.actor_lr=1e-5 \
        algorithm.l2_coef=1e-3 \
        total_epoch=150000 update_step_per_epoch=1 interact_per_epoch=0 \
        test_interval=2000 test_episodes=10 \
        save_interval=5000 train_log_interval=1000 \
        expert_dataset.data_path="${PWD}/outputs/collected/SAC-Mujoco/sac_${env_name}_newest_100eps_min_reward_0.hdf5" \
        expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 \
        use_swanlab=false use_tensorboard=true \
        "$@"
done
