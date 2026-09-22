#!/usr/bin/env bash
set -e

ROOT="/home/ubuntu/wangchenyang/rlzero/rlfromscratch"
COLLECT_PY="${ROOT}/collect.py"
RUN_ROOT="${ROOT}/outputs/PPO-search/2026-07-19_12-03-45"
OUTPUT_ROOT="${ROOT}/outputs/imitation/dataset/ppo_epoch2000_collected"
VIEW_ROOT="${OUTPUT_ROOT}/.checkpoint_views"

source "/home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh"
conda activate rlzero

mkdir -p "${OUTPUT_ROOT}" "${VIEW_ROOT}"

# 这批 PPO 训练的 total_epoch=3000、save_interval=1000。
# 训练循环会在 epoch 0/1000/2000 覆盖 newest_model.pth，
# 因此下面指定的 newest_model.pth 就是 epoch 2000 checkpoint。
collect() {
    env_name="$1"
    run_dir="$2"

    config_path="${run_dir}/.hydra/config.yaml"
    if [ ! -f "${run_dir}/newest_model.pth" ] || [ ! -f "${config_path}" ]; then
        echo "Missing epoch-2000 checkpoint or config: ${run_dir}" >&2
        exit 1
    fi
    if ! grep -Eq '^total_epoch:[[:space:]]*3000$' "${config_path}" || \
       ! grep -Eq '^save_interval:[[:space:]]*1000$' "${config_path}"; then
        echo "Unexpected save schedule; cannot identify newest_model.pth as epoch 2000: ${run_dir}" >&2
        exit 1
    fi

    # 7 月 19 日的旧运行把对应的归一化统计保存成 obs_rms.pth，
    # 当前 collect.py 则按 --model newest 查找 newest_obs_rms.pth。
    # 用只包含软链接的视图兼容旧命名，不修改原训练目录。
    view_dir="${VIEW_ROOT}/${env_name}"
    mkdir -p "${view_dir}/.hydra"
    ln -sfn "${run_dir}/newest_model.pth" "${view_dir}/newest_model.pth"
    ln -sfn "${config_path}" "${view_dir}/.hydra/config.yaml"
    if [ -f "${run_dir}/newest_obs_rms.pth" ]; then
        ln -sfn "${run_dir}/newest_obs_rms.pth" "${view_dir}/newest_obs_rms.pth"
    elif [ -f "${run_dir}/obs_rms.pth" ]; then
        ln -sfn "${run_dir}/obs_rms.pth" "${view_dir}/newest_obs_rms.pth"
    else
        echo "Missing observation-normalization statistics: ${run_dir}" >&2
        exit 1
    fi

    echo "===== Collecting PPO ${env_name} from epoch 2000 ====="
    python "${COLLECT_PY}" \
        --run-dir "${view_dir}" \
        --model newest \
        --output "${OUTPUT_ROOT}/ppo_${env_name}_epoch2000_100eps_min_reward_0" \
        --collect-episodes 100 \
        --save-every 0 \
        --num-testing-envs 8 \
        --seed 0 \
        --min-reward 0 \
        --overwrite \
        --min-length 1000 \
        --test-epsilon 0.0
}

# 在 epoch 2000 的测试记录中，seed 2 是这四个环境各自的最高分种子。
collect "HalfCheetah-v5" "${RUN_ROOT}/seed-2_name-HalfCheetah-v5"
collect "Hopper-v5"      "${RUN_ROOT}/seed-2_name-Hopper-v5"
collect "Walker2d-v5"    "${RUN_ROOT}/seed-2_name-Walker2d-v5"
collect "Ant-v5"         "${RUN_ROOT}/seed-2_name-Ant-v5"

echo "===== Finished all PPO-MuJoCo epoch-2000 collections ====="
echo "Saved under: ${OUTPUT_ROOT}"
