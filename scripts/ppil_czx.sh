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
    # Local deep_PPIL SAC/environment defaults; README overrides below.
    actor_lr=3e-4
    init_temp=0.01
    tau=1
    betas='[0.9,0.999]'
    weight_source=online
    loss_type=value
    regularize=false
    use_target=true
    target_entropy=-1.0
    case "$env_name" in
        # README Ant inherits agent.online=true from the local SAC config.
        Ant-v5)
            actor_lr=3e-5; loss_type=value_expert; regularize=true; use_target=false ;;
        # README HalfCheetah explicitly selects agent.online=false.
        HalfCheetah-v5)
            actor_lr=3e-5; tau=0.08; weight_source=expert
            loss_type=value_expert; regularize=true; use_target=false ;;
        # No README PPIL commands: use local SAC + environment YAML defaults.
        Hopper-v5|Walker2d-v5) ;;
        # No local Humanoid reference: retain ppil.sh settings.
        Humanoid-v5)
            actor_lr=3e-5; tau=0.08; weight_source=expert
            loss_type=value_expert; regularize=true; use_target=false
            target_entropy=-1.916290731874155 ;;
    esac
    python run_ppil.py \
        random_steps=0 \
        algorithm=ppil env=mujoco env.name="$env_name" \
        env.num_training_envs=1 env.num_testing_envs=1 \
        env.obs_norm=false env.absorbing=false \
        seed=0 gamma=0.99 total_epoch=500000 \
        interact_per_epoch=1 update_step_per_epoch=1 start_train_step=1280 \
        train_action_deterministic=false test_episodes=10 \
        test_interval=10000 save_interval=100000 train_log_interval=1000 \
        algorithm.weight_source="$weight_source" algorithm.loss_type="$loss_type" \
        algorithm.tau="$tau" algorithm.inner_steps=20 \
        algorithm.regularize="$regularize" algorithm.chi=false \
        algorithm.r_reg_coef=0.5 algorithm.q_reg_coef=0.5 \
        algorithm.rl.critic_lr=3e-4 algorithm.reward_lr=3e-4 \
        algorithm.rl.double_critic=false algorithm.rl.critic_loss_coef=1.0 \
        algorithm.rl.target_update_method=soft algorithm.rl.target_update_tau=0.005 \
        algorithm.rl.target_update_interval=1 \
        algorithm.rl.use_target="$use_target" algorithm.rl.learn_temp=false \
        algorithm.rl.init_temp="$init_temp" algorithm.rl.actor_lr="$actor_lr" \
        algorithm.actor_betas="$betas" algorithm.critic_betas="$betas" \
        algorithm.reward_betas="$betas" \
        algorithm.rl.target_entropy="$target_entropy" \
        expert_dataset.data_path="${PWD}/outputs/collected/SAC-Mujoco/sac_${env_name}_newest_100eps_min_reward_0.hdf5" \
        expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 \
        use_swanlab=false use_tensorboard=true \
        "${extra_args[@]}"
done
