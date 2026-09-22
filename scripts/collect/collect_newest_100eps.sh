#!/usr/bin/env bash
set -e

COLLECT_PY="/home/ubuntu/wangchenyang/rlzero/rlfromscratch/collect.py"
OUTPUT_ROOT="/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/collected"

mkdir -p "${OUTPUT_ROOT}"

# 参数依次是：保存文件夹、文件名前缀、环境、log/run 文件夹、自定义 min_reward。
# 第 6 个参数 save_every 可选，默认 0（不分段）；Atari 使用 10。
# 没有填写的任务会自动跳过。
# --overwrite sets overwrite=True and replaces existing data at the same output path.
# Enabled: 20 completed MuJoCo reruns from 2026-08-24_15-51-00.
# Atari calls remain commented: new reruns are incomplete or unavailable.
collect() {
    output_folder="$1"
    algorithm_name="$2"
    env_name="$3"
    run_dir="$4"
    custom_min_reward="$5"
    save_every="${6:-0}"

    if [ -z "${run_dir}" ] || [ -z "${custom_min_reward}" ]; then
        return
    fi

    output_dir="${OUTPUT_ROOT}/${output_folder}"
    mkdir -p "${output_dir}"

    # python "${COLLECT_PY}" \
    #     --run-dir "${run_dir}" \
    #     --model newest \
    #     --output "${output_dir}/${algorithm_name}_${env_name}_newest_100eps_min_reward_0" \
    #     --collect-episodes 100 \
    #     --save-every "${save_every}" \
    #     --num-testing-envs 8 \
    #     --seed 0 \
    #     --min-reward 0 \
    #     --overwrite \
    #     --min-length 1000 \
    #     --test-epsilon 0.0

    python "${COLLECT_PY}" \
        --run-dir "${run_dir}" \
        --model newest \
        --output "${output_dir}/${algorithm_name}_${env_name}_newest_100eps_min_reward_${custom_min_reward}" \
        --collect-episodes 100 \
        --save-every "${save_every}" \
        --num-testing-envs 8 \
        --seed 0 \
        --min-reward "${custom_min_reward}" \
        --overwrite \
        --min-length 0 \
        --test-epsilon 0.001
}

# 格式：
# collect "保存文件夹" "文件名前缀" "环境" "log/run文件夹" "自定义min_reward" [save_every]

# DQN / Atari
# New DQN checkpoints exist, but these reruns did not finish; kept disabled.
# collect "DQN-Atari" "dqn" "BreakoutNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/DQN-best-seed/2026-08-24_15-51-00/seed-5_name-BreakoutNoFrameskip-v4_nstep-1" "300" "10"
# collect "DQN-Atari" "dqn" "SeaquestNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/DQN-best-seed/2026-08-24_15-51-00/seed-8_name-SeaquestNoFrameskip-v4_nstep-1" "5000" "10"
# collect "DQN-Atari" "dqn" "QbertNoFrameskip-v4" "/home/ubuntu/wangchenyang/RLfromscratch/outputs/DQN-search/2026-07-14_20-42-19/seed-7_name-QbertNoFrameskip-v4_nstep-1/" "10000" "10"
# collect "DQN-Atari" "dqn" "SpaceInvadersNoFrameskip-v4" "/home/ubuntu/wangchenyang/RLfromscratch/outputs/DQN-search/2026-07-14_20-42-19/seed-3_name-SpaceInvadersNoFrameskip-v4_nstep-1/" "2000" "10"
# collect "DQN-Atari" "dqn" "BeamRiderNoFrameskip-v4" "/home/ubuntu/wangchenyang/RLfromscratch/outputs/DQN-search/2026-07-14_20-42-19/seed-5_name-BeamRiderNoFrameskip-v4_nstep-1/" "8000" "10"
# collect "DQN-Atari" "dqn" "PongNoFrameskip-v4" "/home/ubuntu/wangchenyang/RLfromscratch/outputs/PongNoFrameskip-v4-DQN/2026-06-28_18-25-47/" "20" "10"
# collect "DQN-Atari" "dqn" "FreewayNoFrameskip-v4" "/home/ubuntu/wangchenyang/RLfromscratch/outputs/DQN-search/2026-07-14_20-42-19/seed-7_name-FreewayNoFrameskip-v4_nstep-1/" "30" "10"

# # PPO / Atari
# collect "PPO-Atari" "ppo" "BreakoutNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-atari-search/2026-07-24_20-44-09/seed-3_name-BreakoutNoFrameskip-v4/" "400" "10"
# collect "PPO-Atari" "ppo" "SeaquestNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-atari-search/2026-07-24_20-44-09/seed-5_name-SeaquestNoFrameskip-v4/" "1000" "10"
# collect "PPO-Atari" "ppo" "QbertNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-atari-search/2026-07-24_20-44-09/seed-9_name-QbertNoFrameskip-v4/" "10000" "10"
# collect "PPO-Atari" "ppo" "SpaceInvadersNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-atari-search/2026-07-24_20-44-09/seed-1_name-SpaceInvadersNoFrameskip-v4/" "1000" "10"
# collect "PPO-Atari" "ppo" "BeamRiderNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-atari-search/2026-07-24_20-44-09/seed-6_name-BeamRiderNoFrameskip-v4/" "3000" "10"
# collect "PPO-Atari" "ppo" "PongNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-atari-search/2026-07-24_20-44-09/seed-8_name-PongNoFrameskip-v4/" "10" "10"
# collect "PPO-Atari" "ppo" "FreewayNoFrameskip-v4" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-atari-search/2026-07-24_20-44-09/seed-1_name-FreewayNoFrameskip-v4/" "30" "10"

# # # SAC / MuJoCo
collect "SAC-Mujoco" "sac" "Ant-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/SAC-best-seed/2026-08-24_15-51-00/seed-1_name-Ant-v5" "6000"
collect "SAC-Mujoco" "sac" "Hopper-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/SAC-best-seed/2026-08-24_15-51-00/seed-4_name-Hopper-v5" "3000"
collect "SAC-Mujoco" "sac" "HalfCheetah-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/SAC-best-seed/2026-08-24_15-51-00/seed-5_name-HalfCheetah-v5" "10000"
collect "SAC-Mujoco" "sac" "Walker2d-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/SAC-best-seed/2026-08-24_15-51-00/seed-5_name-Walker2d-v5" "5000"
collect "SAC-Mujoco" "sac" "Humanoid-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/SAC-best-seed/2026-08-24_15-51-00/seed-8_name-Humanoid-v5" "5000"

# # # TD3 / MuJoCo
collect "TD3-Mujoco" "td3" "Ant-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TD3-best-seed/2026-08-24_15-51-00/seed-5_name-Ant-v5" "5000"
collect "TD3-Mujoco" "td3" "Hopper-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TD3-best-seed/2026-08-24_15-51-00/seed-6_name-Hopper-v5" "3000"
collect "TD3-Mujoco" "td3" "HalfCheetah-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TD3-best-seed/2026-08-24_15-51-00/seed-5_name-HalfCheetah-v5" "10000"
collect "TD3-Mujoco" "td3" "Walker2d-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TD3-best-seed/2026-08-24_15-51-00/seed-5_name-Walker2d-v5" "4000"
collect "TD3-Mujoco" "td3" "Humanoid-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TD3-best-seed/2026-08-24_15-51-00/seed-9_name-Humanoid-v5" "4000"

# # PPO / MuJoCo (obs_norm=true, latest completed best-seed reruns)
collect "PPO-Mujoco" "ppo" "Ant-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-3_name-Ant-v5" "4000"
collect "PPO-Mujoco" "ppo" "Hopper-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-1_name-Hopper-v5" "3000"
collect "PPO-Mujoco" "ppo" "HalfCheetah-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-9_name-HalfCheetah-v5" "5000"
collect "PPO-Mujoco" "ppo" "Walker2d-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-0_name-Walker2d-v5" "5000"
collect "PPO-Mujoco" "ppo" "Humanoid-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/PPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-3_name-Humanoid-v5" "5000"

# TRPO / MuJoCo (obs_norm=true, latest completed best-seed reruns)
collect "TRPO-Mujoco" "trpo" "Ant-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TRPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-0_name-Ant-v5" "5000"
collect "TRPO-Mujoco" "trpo" "Hopper-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TRPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-4_name-Hopper-v5" "3000"
collect "TRPO-Mujoco" "trpo" "HalfCheetah-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TRPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-8_name-HalfCheetah-v5" "8000"
collect "TRPO-Mujoco" "trpo" "Walker2d-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TRPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-5_name-Walker2d-v5" "5000"
collect "TRPO-Mujoco" "trpo" "Humanoid-v5" "/home/ubuntu/wangchenyang/rlzero/rlfromscratch/outputs/TRPO-best-seed-obsnorm/2026-08-24_15-51-00/seed-2_name-Humanoid-v5" "4000"
