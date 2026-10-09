#!/usr/bin/env bash
set -euo pipefail

# 参数对齐到当前 garage MuJoCo-v5 入口；其余配置复用 hype.sh，含 5 条专家轨迹。
# 默认依次运行五个环境；例：bash scripts/hype_garage.sh env.name=HalfCheetah-v5
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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

# Actor 共享隐藏层；Actor/Critic 使用默认 Linear 初始化，D 独立采用
# gain=1 的正交权重初始化并保留默认 bias；双 Q 的 MSE 之和乘以 0.5。
# 环境扰动/阶段重置、动作与熵坐标仍未完全对齐；保留 Humanoid 的熵尺度修正。
# 每轮 LR 分母仍是实际采集步数，而 garage 使用名义步数；当前主循环也不会
# 在最后一轮结束后额外评估。这些均需改训练代码，不能仅用启动参数解决。
for env_name in "${env_names[@]}"; do
    nominal_phase_steps=10000
    if [[ "$env_name" == Hopper-v5 ]]; then nominal_phase_steps=5000; fi
    # garage 每次 learn 按 64 步收集，实际每轮为 10048 / 5056 步。
    # 100 / 200 个完整轮次；每 5 / 10 轮评估一次，对应名义 50000 步。
    phase_epochs=$(((nominal_phase_steps + 63) / 64))
    total_epochs=$((1000000 / nominal_phase_steps * phase_epochs))
    eval_epochs=$((50000 / nominal_phase_steps * phase_epochs))

    bash "$SCRIPT_DIR/hype.sh" \
        env.name="$env_name" \
        experiment_name="${env_name}-HyPESAC-garage" \
        algorithm.reset_policy_timesteps=true \
        algorithm.policy_lr_decay=true \
        start_train_step=100 \
        algorithm.rl.initialize=false \
        algorithm.rl.common_net=true \
        algorithm.rl.critic_loss_coef=0.5 \
        algorithm.discriminator.discriminator_initialization=orthogonal \
        algorithm.discriminator.discriminator_output_gain=1.0 \
        discriminator_train_interval="$phase_epochs" \
        total_epoch="$total_epochs" \
        test_interval="$eval_epochs" \
        "${extra_args[@]}"
done
