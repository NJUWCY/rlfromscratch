#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use the same five v5 tasks and expert datasets as scripts/hype.sh.
# By default train sequentially; env.name=... selects one environment.
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

for env_name in "${env_names[@]}"; do
    target_entropy=-1.0
    if [[ "$env_name" == Humanoid-v5 ]]; then
        # Per dimension: -1 + log(0.4), consistent with the scaled actor log-prob.
        target_entropy=-1.916290731874155
    fi
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
    printf '[SQIL] environment=%s expert_data=%s\n' "$env_name" "$expert_data_path"

    # 1M interactions, 64 transitions / 64 updates per epoch, 5 complete demos.
    python run_sqil.py \
        random_steps='${start_train_step}' \
        algorithm=sqil \
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
        start_train_step=1000 \
        train_action_deterministic=false \
        test_interval=782 \
        test_episodes=10 \
        train_log_interval=16 \
        save_interval=1563 \
        algorithm.rl.target_entropy="${target_entropy}" \
        expert_dataset.data_path="${expert_data_path}" \
        expert_dataset.trajectory_num=5 \
        expert_dataset.subsample_frequency=1 \
        use_swanlab=false \
        use_tensorboard=true \
        "${extra_args[@]}"
done
