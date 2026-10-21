#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

for seed in 1 2; do
    python run_pwil.py \
        random_steps='${start_train_step}' \
            algorithm=pwil env=mujoco env.name=Ant-v5 \
            env.num_training_envs=1 env.num_testing_envs=1 \
            env.obs_norm=false env.absorbing=false \
            seed=0 gamma=0.99 \
            total_epoch=15625 interact_per_epoch=64 update_step_per_epoch=64 \
            start_train_step=1000 train_action_deterministic=false \
            test_interval=156 test_episodes=10 \
            save_interval=1563 train_log_interval=16 \
            algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.01 \
            algorithm.rl.target_entropy=-1 \
            algorithm.alpha=1 \
            expert_dataset.data_path="${PWD}/outputs/collected/SAC-Mujoco/sac_Ant-v5_newest_100eps_min_reward_0.hdf5" \
            expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 \
            use_swanlab=false use_tensorboard=true \
            seed=${seed} \
            "$@"
done