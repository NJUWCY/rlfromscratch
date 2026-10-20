#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

env_names=(Hopper-v5 Ant-v5)
extra_args=()
for arg in "$@"; do
    case "$arg" in
        env.name=*) env_names=("${arg#env.name=}") ;;
        *) extra_args+=("$arg") ;;
    esac
done
for env_name in "${env_names[@]}"; do
    case "$env_name" in
        Hopper-v5|Ant-v5) ;;
        *) printf 'Unsupported environment: %s\n' "$env_name" >&2; exit 1 ;;
    esac
done
for env_name in "${env_names[@]}"; do
    case "$env_name" in
        Hopper-v5)
            # Reference: outputs/Hopper-v5-IQLearn/2026-10-18_06-21-13/.hydra/config.yaml
            # Zero bounds and penalty=0.5 preserve the original squared regularizer.
            env_args=(
                "algorithm.iq_loss_type=v0"
                "algorithm.left=0"
                "algorithm.penalty=0.5"
                "algorithm.right=0"
                "algorithm.rl.actor_lr=3e-05"
                "algorithm.rl.init_temp=0.1"
                "env.name=Hopper-v5"
                "expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/sac_Hopper-v5_newest_100eps_min_reward_0.hdf5"
            ) ;;
        Ant-v5)
            # Reference: outputs/IQLearn-Ant-v5-search/2026-10-18_15-03-45/init-temp-0.01_actor-lr-0.0001_left-0.1_right-10_penalty-1/.hydra/config.yaml
            env_args=(
                "algorithm.iq_loss_type=value"
                "algorithm.left=-0.1"
                "algorithm.penalty=1"
                "algorithm.right=10"
                "algorithm.rl.actor_lr=0.0001"
                "algorithm.rl.init_temp=0.01"
                "env.name=Ant-v5"
                "expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/sac_Ant-v5_newest_100eps_min_reward_0.hdf5"
            ) ;;
    esac
    python run_iqlearn.py \
        algorithm=iqlearn \
        env=mujoco \
        "env.num_training_envs=1" \
        "env.num_testing_envs=1" \
        "env.obs_norm=false" \
        "env.absorbing=false" \
        "env.collect_init_states=true" \
        "gamma=0.99" \
        "seed=0" \
        "use_tensorboard=true" \
        "use_swanlab=false" \
        "save_interval=100000" \
        "test_interval=10000" \
        "train_log_interval=10000" \
        "total_epoch=1000000" \
        "interact_per_epoch=1" \
        "update_step_per_epoch=1" \
        "test_episodes=10" \
        "train_action_deterministic=false" \
        "start_train_step=5000" \
        "expert_dataset.trajectory_num=5" \
        "expert_dataset.subsample_frequency=1" \
        "algorithm.rl.critic_lr=0.0003" \
        "algorithm.rl.learn_temp=false" \
        "algorithm.v0_mix_type=mix" \
        "algorithm.regularize=true" \
        "algorithm.alpha=0.5" \
        "${env_args[@]}" "${extra_args[@]}"
done
