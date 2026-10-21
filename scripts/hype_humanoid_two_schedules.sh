#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export SWANLAB_MODE=online PYTHONUNBUFFERED=1
unset SWANLAB_CORE_SKIP_STORE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TQDM_MININTERVAL=60

# Reuse the credential from the existing script; never execute its training commands.
if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    source <(sed -n '/^[[:space:]]*export SWANLAB_API_KEY=/p' scripts/dac.sh)
fi

# Schedule endpoints count completed SAC updates.
schedule_a='[[0.025,0.0,500000],[0.0,0.0,1500032]]'
schedule_b='[[0.2,0.1,100000],[0.1,0.0,500000],[0.0,0.0,1500032]]'
run_root="${HYPE_LOG_ROOT:-outputs/HyPESAC-Humanoid-two-expert-schedules/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$run_root/A_fast_OAdam" "$run_root/A_fast_Adam" "$run_root/A_mid_fixed" "$run_root/A_slow_fixed" "$run_root/B_fast_OAdam" "$run_root/B_fast_Adam" "$run_root/B_mid_fixed" "$run_root/B_slow_fixed"

# 1. Schedule A: fast_OAdam
CUDA_VISIBLE_DEVICES=0 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0003 algorithm.rl.critic_lr=0.0003 \
    algorithm.discriminator.disc_lr=0.0003 algorithm.discriminator.discriminator_optimizer=OAdam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-0.5 \
    algorithm.sampling_schedule="$schedule_a" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/A_fast_OAdam" "$@" >"$run_root/A_fast_OAdam/stdout.log" 2>&1 &
p1=$!

# 2. Schedule A: fast_Adam
CUDA_VISIBLE_DEVICES=1 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0003 algorithm.rl.critic_lr=0.0003 \
    algorithm.discriminator.disc_lr=3e-05 algorithm.discriminator.discriminator_optimizer=Adam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-0.5 \
    algorithm.sampling_schedule="$schedule_a" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/A_fast_Adam" "$@" >"$run_root/A_fast_Adam/stdout.log" 2>&1 &
p2=$!

# 3. Schedule A: mid_fixed
CUDA_VISIBLE_DEVICES=0 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0003 algorithm.rl.critic_lr=0.0001 \
    algorithm.discriminator.disc_lr=0.0001 algorithm.discriminator.discriminator_optimizer=OAdam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-0.5 \
    algorithm.sampling_schedule="$schedule_a" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/A_mid_fixed" "$@" >"$run_root/A_mid_fixed/stdout.log" 2>&1 &
p3=$!

# 4. Schedule A: slow_fixed
CUDA_VISIBLE_DEVICES=1 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0001 algorithm.rl.critic_lr=0.0001 \
    algorithm.discriminator.disc_lr=3e-05 algorithm.discriminator.discriminator_optimizer=Adam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-1.0 \
    algorithm.sampling_schedule="$schedule_a" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/A_slow_fixed" "$@" >"$run_root/A_slow_fixed/stdout.log" 2>&1 &
p4=$!

# 5. Schedule B: fast_OAdam
CUDA_VISIBLE_DEVICES=0 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0003 algorithm.rl.critic_lr=0.0003 \
    algorithm.discriminator.disc_lr=0.0003 algorithm.discriminator.discriminator_optimizer=OAdam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-0.5 \
    algorithm.sampling_schedule="$schedule_b" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/B_fast_OAdam" "$@" >"$run_root/B_fast_OAdam/stdout.log" 2>&1 &
p5=$!

# 6. Schedule B: fast_Adam
CUDA_VISIBLE_DEVICES=1 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0003 algorithm.rl.critic_lr=0.0003 \
    algorithm.discriminator.disc_lr=3e-05 algorithm.discriminator.discriminator_optimizer=Adam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-0.5 \
    algorithm.sampling_schedule="$schedule_b" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/B_fast_Adam" "$@" >"$run_root/B_fast_Adam/stdout.log" 2>&1 &
p6=$!

# 7. Schedule B: mid_fixed
CUDA_VISIBLE_DEVICES=0 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0003 algorithm.rl.critic_lr=0.0001 \
    algorithm.discriminator.disc_lr=0.0001 algorithm.discriminator.discriminator_optimizer=OAdam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-0.5 \
    algorithm.sampling_schedule="$schedule_b" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/B_mid_fixed" "$@" >"$run_root/B_mid_fixed/stdout.log" 2>&1 &
p7=$!

# 8. Schedule B: slow_fixed
CUDA_VISIBLE_DEVICES=1 python run_hype.py \
    random_steps='${start_train_step}' \
    algorithm.rl.actor_lr=0.0001 algorithm.rl.critic_lr=0.0001 \
    algorithm.discriminator.disc_lr=3e-05 algorithm.discriminator.discriminator_optimizer=Adam \
    algorithm.rl.learn_temp=false algorithm.rl.init_temp=0.1 algorithm.rl.target_entropy=-1.0 \
    algorithm.sampling_schedule="$schedule_b" \
    algorithm=hypesac env=mujoco env.name=Humanoid-v5 env.num_training_envs=1 \
    env.num_testing_envs=8 env.obs_norm=false env.scale=false env.absorbing=true \
    env.max_episode_length=1000 test_epsilon=0 gamma=0.99 seed=0 \
    use_wandb=false use_tensorboard=true use_swanlab=true save_interval=1563 \
    test_interval=157 train_log_interval=16 reward_buffer_size=20 total_epoch=23438 \
    interact_per_epoch=64 update_step_per_epoch=64 update_discriminator_step_per_epoch=1 discriminator_train_interval=157 \
    test_episodes=10 train_action_deterministic=false start_train_step=1000 expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/humanoid_wave1_fixed5.hdf5 \
    expert_dataset.trajectory_num=5 expert_dataset.subsample_frequency=1 expert_dataset.random_start=false algorithm.rl.name=SAC \
    algorithm.rl.onpolicy=false algorithm.rl.buffer_name=AbsorbingReplayBuffer algorithm.rl.buffer_size=1000000 algorithm.rl.temp_lr=0.0003 \
    algorithm.rl.critic_optimizer=Adam algorithm.rl.actor_optimizer=Adam algorithm.rl.temp_optimizer=Adam algorithm.rl.double_critic=true \
    algorithm.rl.use_target=true algorithm.rl.target_update_method=soft algorithm.rl.target_update_interval=1 algorithm.rl.target_update_tau=0.01 \
    algorithm.rl.batch_size=256 algorithm.rl.critic_loss_coef=1.0 algorithm.rl.hidden_sizes='[256,256]' algorithm.rl.activation=relu \
    algorithm.rl.rescale=true algorithm.rl.state_dependent_std=true algorithm.rl.action_bound_method=tanh algorithm.rl.update_log_temp=true \
    algorithm.rl.nstep=1 algorithm.rl.initialize=true algorithm.discriminator.name=HyPE algorithm.discriminator.discriminator_hidden_sizes='[256,256]' \
    algorithm.discriminator.discriminator_activation=tanh algorithm.discriminator.spectral_norm=true algorithm.discriminator.discriminator_weight_decay=0.0 algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_gradient_penalty=true algorithm.discriminator.discriminator_train_steps=1 algorithm.discriminator.discriminator_batch_size=4096 algorithm.discriminator.reward_function=HyPE \
    algorithm.discriminator.target_gradient_norm=1.0 algorithm.policy_lr_decay=false algorithm.reset_policy_timesteps=false algorithm.discriminator.disc_lr_decay=false \
    algorithm.discriminator.discriminator_initialization=orthogonal_relu algorithm.discriminator.discriminator_output_gain=0.01 algorithm.rl.common_net=false algorithm.discriminator.rollout_episodes=4 \
    log_dir="$run_root/B_slow_fixed" "$@" >"$run_root/B_slow_fixed/stdout.log" 2>&1 &
p8=$!

# Wait for every job and propagate any failure.
status=0
for pid in "$p1" "$p2" "$p3" "$p4" "$p5" "$p6" "$p7" "$p8"; do
    wait "$pid" || status=1
done
exit "$status"
