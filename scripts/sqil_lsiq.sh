#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

env_names=(Ant-v5 HalfCheetah-v5 Hopper-v5 Walker2d-v5 Humanoid-v5)
for arg in "$@"; do
    if [[ "$arg" == env.name=* ]]; then env_names=("${arg#env.name=}"); fi
done

source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

for env_name in "${env_names[@]}"; do
    target_entropy=-1.0
    if [[ "$env_name" == Humanoid-v5 ]]; then target_entropy=-1.916290731874155; fi
    expert_data_path="/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/sac_${env_name}_newest_100eps_min_reward_0.hdf5"

    python run_sqil.py \
        random_steps='${start_train_step}' \
        algorithm=sqil \
        env=mujoco \
        env.name="${env_name}" \
        experiment_name="${env_name}-SQIL-LSIQ" \
        env.num_training_envs=1 \
        env.num_testing_envs=1 \
        env.obs_norm=false \
        env.absorbing=false \
        seed=0 \
        gamma=0.98 \
        total_epoch=15625 \
        interact_per_epoch=64 \
        update_step_per_epoch=64 \
        start_train_step=1000 \
        train_action_deterministic=false \
        test_interval=782 \
        test_episodes=10 \
        train_log_interval=16 \
        save_interval=1563 \
        algorithm.rl.double_critic=false \
        algorithm.rl.critic_loss_coef=0.5 \
        algorithm.rl.learn_temp=false \
        algorithm.rl.init_temp=0.001 \
        algorithm.rl.update_log_temp=false \
        algorithm.rl.batch_size=256 \
        algorithm.sampling_schedule='[[0.5,0.5,1000000]]' \
        algorithm.rl.target_entropy="${target_entropy}" \
        expert_dataset.data_path="${expert_data_path}" \
        expert_dataset.trajectory_num=5 \
        expert_dataset.subsample_frequency=1 \
        use_swanlab=false \
        use_tensorboard=true \
        "$@"
done
