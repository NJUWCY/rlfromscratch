#!/usr/bin/env bash
set -e

cd /home/ubuntu/wangchenyang/rlzero/rlfromscratch/

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-sac}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY=yyKpLHGppV78RFW0p1PNQ

for env_name in Ant-v5; do
    python run_gail.py \
        algorithm=gailppo \
        env=mujoco \
        env.name="${env_name}" \
        algorithm.discriminator.discriminator_train_steps=8 \
        algorithm.discriminator.discriminator_expert_batch_size=128 \
        algorithm.discriminator.discriminator_policy_batch_size=128 \
        env.num_training_envs=8 \
        env.num_testing_envs=8 \
        seed=0 \
        total_epoch=3000 \
        train_log_interval=10 \
        save_interval=1000 \
        test_interval=5 \
        train_action_deterministic=false \
        algorithm.rl.collect_traj=false \
        algorithm.rl.buffer_name=ReplayBuffer \
        algorithm.rl.update_epochs=5 \
        algorithm.rl.minibatch_size=128 \
        algorithm.rl.use_grad_clip=true \
        algorithm.rl.advan_norm=false \
        env.obs_norm=true \
        algorithm.rl.rescale=true \
        algorithm.rl.return_scaling=true \
        algorithm.rl.lr_decay=true \
        interact_per_epoch=128 \
        algorithm.rl.initialize=true \
        algorithm.rl.recompute_adv=false \
        algorithm.rl.initial_log_sigma=-0.5 \
        start_train_step=0 \
        expert_dataset.data_path="/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/SAC-Mujoco/sac_${env_name}_newest_100eps_min_reward_0.hdf5" \
        expert_dataset.trajectory_num=10 \
        expert_dataset.subsample_frequency=1 \
        "$@"
done
