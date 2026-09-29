set -e

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY=yyKpLHGppV78RFW0p1PNQ
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# DAC: off-policy adversarial imitation (GAIL discriminator + SAC + absorbing states).
# One environment, one gradient step per environment step, as in the DAC paper.
for env in Humanoid-v5; do
    python run_dac.py \
        algorithm=dacsac \
        env=mujoco \
        env.name=${env} \
        env.num_training_envs=1 \
        env.num_testing_envs=8 \
        env.obs_norm=false \
        env.absorbing=true \
        seed=0 \
        gamma=0.97 \
        total_epoch=1500000 \
        interact_per_epoch=1 \
        update_step_per_epoch=1 \
        update_discriminator_step_per_epoch=1 \
        discriminator_train_interval=1 \
        start_train_step=1000 \
        test_interval=5000 \
        test_episodes=10 \
        train_log_interval=1000 \
        save_interval=100000 \
        train_action_deterministic=false \
        algorithm.rl.buffer_size=1000000 \
        algorithm.rl.batch_size=256 \
        algorithm.rl.actor_lr=3e-4 \
        algorithm.rl.critic_lr=3e-4 \
        algorithm.rl.temp_lr=3e-4 \
        algorithm.rl.init_temp=1.0 \
        algorithm.rl.target_entropy=-1.4 \
        algorithm.rl.hidden_sizes=[256,256] \
        algorithm.rl.update_log_temp=false \
        algorithm.discriminator.disc_lr=3e-5 \
        algorithm.discriminator.discriminator_optimizer=AdamW \
        algorithm.discriminator.discriminator_weight_decay=10.0 \
        algorithm.discriminator.discriminator_hidden_sizes=[64] \
        algorithm.discriminator.discriminator_activation=relu \
        algorithm.discriminator.discriminator_train_steps=1 \
        algorithm.discriminator.discriminator_batch_size=256 \
        algorithm.discriminator.gradient_penalty_coef=1.0 \
        expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/sac_${env}_newest_100eps_min_reward_0.hdf5 \
        expert_dataset.trajectory_num=10 \
        expert_dataset.subsample_frequency=1 \
        algorithm.discriminator.reward_function=AIRL \
        "$@"
done
