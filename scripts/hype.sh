#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Train all five tasks sequentially, or select one with env.name=Hopper-v5.
# Other Hydra overrides are applied to every selected task.
env_names=(Ant-v5 HalfCheetah-v5 Hopper-v5 Walker2d-v5 Humanoid-v5)
extra_args=()
for arg in "$@"; do
    case "$arg" in
        env.name=*) env_names=("${arg#env.name=}") ;;
        *) extra_args+=("$arg") ;;
    esac
done
for env_name in "${env_names[@]}"; do
    case "$env_name" in
        Ant-v5|HalfCheetah-v5|Hopper-v5|Walker2d-v5|Humanoid-v5) ;;
        *) printf 'Unsupported environment: %s\n' "$env_name" >&2; exit 1 ;;
    esac
done

source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Each epoch collects 64 transitions and performs 64 SAC updates.
# Intervals use epochs: ceil(10000/64)=157; ceil(5000/64)=79 for Hopper.
# Target entropy is per dimension, expressed in environment action units.
for env_name in "${env_names[@]}"; do
    [[ "$env_name" == Humanoid-v5 ]] && continue  # Dedicated command below.
    target_entropy=-1.0
    discriminator_interval=157
    sampling_schedule='[[0.2,0.1,200000],[0.1,0.01,1000000]]'
    env_overrides=()
    case "$env_name" in
        Ant-v5|Walker2d-v5)
            sampling_schedule='[[0.2,0.2,300000],[0.2,0.1,1000000]]'
            ;;
        Hopper-v5)
            discriminator_interval=79
            ;;
        HalfCheetah-v5)
            # Best round1 configuration across seeds 0, 1, 2.
            sampling_schedule='[[0.025,0.025,1000000]]'
            env_overrides=(
                algorithm.rl.actor_optimizer=Adam
                algorithm.rl.critic_optimizer=Adam
                algorithm.rl.target_update_tau=0.02
            )
            ;;
    esac
    expert_data_path="${PWD}/outputs/collected/SAC-Mujoco/sac_${env_name}_newest_100eps_min_reward_0.hdf5"
    for arg in "${extra_args[@]}"; do
        case "$arg" in
            expert_dataset.data_path=*) expert_data_path="${arg#expert_dataset.data_path=}" ;;
        esac
    done
    if [[ ! -f "$expert_data_path" ]]; then
        printf 'Expert dataset not found for %s: %s\n' "$env_name" "$expert_data_path" >&2
        exit 1
    fi
    printf '[HyPE] environment=%s expert_data=%s\n' "$env_name" "$expert_data_path"

    python run_hype.py \
        random_steps='${start_train_step}' \
        algorithm=hypesac \
        env=mujoco \
        env.name="${env_name}" \
        env.num_training_envs=1 \
        env.num_testing_envs=1 \
        env.obs_norm=false \
        env.absorbing=false \
        seed=0 \
        gamma=0.98 \
        total_epoch=15625 \
        interact_per_epoch=64 \
        update_step_per_epoch=64 \
        update_discriminator_step_per_epoch=1 \
        discriminator_train_interval="${discriminator_interval}" \
        start_train_step=1000 \
        train_action_deterministic=false \
        test_interval=782 \
        test_episodes=10 \
        train_log_interval=16 \
        save_interval=1563 \
        algorithm.rl.target_entropy="${target_entropy}" \
        algorithm.sampling_schedule="${sampling_schedule}" \
        expert_dataset.data_path="${expert_data_path}" \
        expert_dataset.trajectory_num=5 \
        expert_dataset.subsample_frequency=1 \
        use_swanlab=false \
        use_tensorboard=true \
        "${env_overrides[@]}" \
        "${extra_args[@]}"
done

# Humanoid-v5: 07_ratio0p2_100k_mid_fixed (w45n1f2f).
if [[ " ${env_names[*]} " == *" Humanoid-v5 "* ]]; then
    python run_hype.py \
        random_steps='${start_train_step}' \
        algorithm=hypesac env=mujoco env.name=Humanoid-v5 \
        env.num_training_envs=1 env.absorbing=true \
        total_epoch=23438 interact_per_epoch=64 update_step_per_epoch=64 \
        discriminator_train_interval=157 start_train_step=1000 \
        train_action_deterministic=false test_interval=157 \
        train_log_interval=16 save_interval=1563 \
        algorithm.rl.actor_lr=3e-4 algorithm.rl.critic_lr=1e-4 \
        algorithm.rl.actor_optimizer=Adam algorithm.rl.critic_optimizer=Adam \
        algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 \
        algorithm.rl.temp_lr=3e-4 algorithm.rl.target_entropy=-0.5 \
        algorithm.rl.target_update_tau=0.01 algorithm.policy_lr_decay=false \
        'algorithm.sampling_schedule=[[0.2,0,100000],[0,0,1500032]]' \
        algorithm.discriminator.disc_lr=1e-4 \
        algorithm.discriminator.discriminator_activation=tanh \
        algorithm.discriminator.discriminator_output_gain=0.01 \
        algorithm.discriminator.spectral_norm=true \
        algorithm.discriminator.target_gradient_norm=1.0 \
        algorithm.discriminator.disc_lr_decay=false \
        expert_dataset.data_path=outputs/collected/SAC-Mujoco/sac_Humanoid-v5_newest_100eps_min_reward_0.hdf5 \
        expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 \
        expert_dataset.random_start=false use_tensorboard=true \
        "${extra_args[@]}"
fi
