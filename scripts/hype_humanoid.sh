#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Reuse the Humanoid HyPE defaults, including 5 complete expert trajectories.
# rescale=false selects DiagGaussianActor; action_bound_method is then unused.
# Gaussian log-probs have no tanh/0.4 Jacobian, so use HyPE's default entropy.
exec bash "${project_dir}/scripts/hype.sh" \
    env.name=Humanoid-v5 \
    algorithm.rl.rescale=false \
    algorithm.rl.target_entropy=-1.0 \
    expert_dataset.data_path="${project_dir}/outputs/collected/PPO-Mujoco/ppo_Humanoid-v5_newest_100eps_min_reward_0.hdf5" \
    experiment_name=Humanoid-v5-HyPESAC-DiagGaussian-PPO \
    "$@"
