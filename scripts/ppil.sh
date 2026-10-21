#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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
for env_name in "${env_names[@]}"; do
    actor_lr=3e-5
    init_temp=0.01
    tau=0.08
    betas='[0.9,0.999]'
    target_entropy=-1.0
    case "$env_name" in
        # deep_PPIL/README.md's Ant setting uses default tau=1.
        Ant-v5) tau=1 ;;
        # Paper Appendix L.7's Hopper/Walker settings.
        Hopper-v5) actor_lr=2e-4; betas='[0.0,0.999]' ;;
        Walker2d-v5) actor_lr=1e-4; init_temp=0.001 ;;
        # Humanoid is a library extension, not a reported PPIL benchmark.
        Humanoid-v5) target_entropy=-1.916290731874155 ;;
    esac
    python run_ppil.py \
        random_steps=0 \
        algorithm=ppil env=mujoco env.name="$env_name" \
        env.num_training_envs=1 env.num_testing_envs=1 \
        env.obs_norm=false env.absorbing=false \
        seed=0 gamma=0.99 total_epoch=1000000 \
        interact_per_epoch=1 update_step_per_epoch=1 start_train_step=1280 \
        train_action_deterministic=false test_episodes=10 \
        test_interval=10000 save_interval=100000 train_log_interval=1000 \
        algorithm.weight_source=expert algorithm.loss_type=value_expert \
        algorithm.tau="$tau" algorithm.inner_steps=20 \
        algorithm.rl.use_target=false algorithm.rl.learn_temp=false \
        algorithm.rl.init_temp="$init_temp" algorithm.rl.actor_lr="$actor_lr" \
        algorithm.actor_betas="$betas" algorithm.critic_betas="$betas" \
        algorithm.reward_betas="$betas" \
        algorithm.rl.target_entropy="$target_entropy" \
        expert_dataset.data_path="${PWD}/outputs/collected/SAC-Mujoco/sac_${env_name}_newest_100eps_min_reward_0.hdf5" \
        expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 \
        use_swanlab=false use_tensorboard=true \
        "${extra_args[@]}"
done
