set -euo pipefail

# Match search/AIRLPPO/airlppo_search.json, using the best HalfCheetah-v5
# setting from 2026-09-09_09-28-06:
# disc_lr=3e-5, obs_norm=true, base=[64,64], potential=[32,32],
# use_rms=false, score_discrim=true, return_scaling=true.
# Five MuJoCo envs; three seeds per env run in parallel, envs sequentially.

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY="${SWANLAB_API_KEY:-yyKpLHGppV78RFW0p1PNQ}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

ENVS=(
    Ant-v5
    HalfCheetah-v5
    Hopper-v5
    Walker2d-v5
    Humanoid-v5
)
SEEDS=(0 1 2)

for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
  python run_airl.py \
    algorithm=airlppo \
    env=mujoco \
    env.name="${env}" \
    env.num_training_envs=8 \
    env.num_testing_envs=8 \
    env.obs_norm=true \
    seed="${seed}" \
    total_epoch=5000 \
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
    algorithm.rl.rescale=false \
    algorithm.rl.return_scaling=true \
    algorithm.rl.lr_decay=true \
    interact_per_epoch=128 \
    algorithm.rl.initialize=true \
    algorithm.rl.recompute_adv=false \
    algorithm.rl.initial_log_sigma=-0.5 \
    algorithm.discriminator.disc_lr=3e-5 \
    algorithm.discriminator.base_hidden_sizes=[64,64] \
    algorithm.discriminator.potential_hidden_sizes=[32,32] \
    algorithm.discriminator.use_action=true \
    algorithm.discriminator.use_rms=false \
    algorithm.discriminator.sub_logprobs=false \
    algorithm.discriminator.gradient_penalty_coef=10.0 \
    algorithm.discriminator.discriminator_train_steps=7 \
    algorithm.discriminator.discriminator_batch_size=128 \
    algorithm.discriminator.discriminator_gradient_penalty=true \
    start_train_step=0 \
    expert_dataset.data_path=/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected/PPO-Mujoco/ppo_${env}_newest_100eps_min_reward_0.hdf5 \
    expert_dataset.trajectory_num=10 \
    expert_dataset.subsample_frequency=10 &
    sleep 60
  done
  wait
done
