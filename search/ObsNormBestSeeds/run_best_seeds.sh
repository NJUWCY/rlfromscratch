#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ubuntu/wangchenyang/rlzero/rlfromscratch"
CONFIG_DIR="${REPO_DIR}/search/ObsNormBestSeeds"

cd "${REPO_DIR}"

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-sac}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=1
export TQDM_MININTERVAL=60
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-$(tr -d '\r\n' < "${REPO_DIR}/api.txt")}"

RUN_TAG=${RUN_TAG:-$(date +"%Y-%m-%d_%H-%M-%S")}
PPO_LOG_DIR="outputs/PPO-best-seed-obsnorm/${RUN_TAG}"
TRPO_LOG_DIR="outputs/TRPO-best-seed-obsnorm/${RUN_TAG}"
SAC_LOG_DIR="outputs/SAC-best-seed/${RUN_TAG}"
TD3_LOG_DIR="outputs/TD3-best-seed/${RUN_TAG}"
DQN_LOG_DIR="outputs/DQN-best-seed/${RUN_TAG}"
PPO_ATARI_LOG_DIR="outputs/PPO-atari-best-seed/${RUN_TAG}"

GPU_IDS=${GPU_IDS:-0,1}
IFS=',' read -r -a GPU_CANDIDATES <<< "${GPU_IDS}"
GPUS=()
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_CANDIDATES[@]}"; do
    gpu_id="${gpu_id//[[:space:]]/}"
    [[ -n "${gpu_id}" ]] || continue
    if [[ -n "${SEEN_GPUS[${gpu_id}]+x}" ]]; then
        echo "ERROR: duplicate GPU id in GPU_IDS: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPUS[${gpu_id}]=1
    GPUS+=("${gpu_id}")
done

if (( ${#GPUS[@]} == 0 )); then
    echo "ERROR: GPU_IDS must contain at least one GPU id" >&2
    exit 2
fi

DRY_RUN=false
for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

if [[ "${DRY_RUN}" == false ]]; then
    mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
    for gpu_id in "${GPUS[@]}"; do
        if ! printf '%s\n' "${AVAILABLE_GPUS[@]}" | grep -Fxq "${gpu_id}"; then
            echo "ERROR: requested GPU ${gpu_id} is not available; detected: ${AVAILABLE_GPUS[*]}" >&2
            exit 2
        fi
    done
fi

run_job() {
    local config_path="$1"
    local log_dir="$2"
    local run_name="$3"
    local gpu_id="$4"
    local expect_obs_rms="$5"
    shift 5

    python parallel_search.py \
        --config "${config_path}" \
        --workers 1 \
        --gpus "${gpu_id}" \
        --log-dir "${log_dir}" \
        "$@"

    [[ "${DRY_RUN}" == false ]] || return 0

    local run_dir="${log_dir}/${run_name}"
    local required_files=(best_model.pth newest_model.pth)
    # Only PPO/TRPO on MuJoCo with obs_norm=true persist observation-norm stats;
    # SAC / TD3 / DQN / PPO-atari do not write obs_rms checkpoints.
    if [[ "${expect_obs_rms}" == "1" ]]; then
        required_files+=(best_obs_rms.pth newest_obs_rms.pth)
    fi
    local required_file
    for required_file in "${required_files[@]}"; do
        if [[ ! -f "${run_dir}/${required_file}" ]]; then
            echo "ERROR: task failed or checkpoint is incomplete: ${run_dir}" >&2
            if [[ -f "${run_dir}/stdout.log" ]]; then
                tail -n 40 "${run_dir}/stdout.log" >&2
            fi
            return 1
        fi
    done
}

# Format: config_file|log_dir|run_name|expect_obs_rms
# expect_obs_rms=1 only for PPO/TRPO MuJoCo experts (obs_norm=true); 0 otherwise.
TASKS=(
    # PPO / MuJoCo (obs_norm=true)
    "ppo_halfcheetah_seed9.json|${PPO_LOG_DIR}|seed-9_name-HalfCheetah-v5|1"
    "ppo_hopper_seed1.json|${PPO_LOG_DIR}|seed-1_name-Hopper-v5|1"
    "ppo_walker2d_seed0.json|${PPO_LOG_DIR}|seed-0_name-Walker2d-v5|1"
    "ppo_ant_seed3.json|${PPO_LOG_DIR}|seed-3_name-Ant-v5|1"
    "ppo_humanoid_seed3.json|${PPO_LOG_DIR}|seed-3_name-Humanoid-v5|1"
    # TRPO / MuJoCo (obs_norm=true)
    "trpo_halfcheetah_seed8.json|${TRPO_LOG_DIR}|seed-8_name-HalfCheetah-v5|1"
    "trpo_hopper_seed4.json|${TRPO_LOG_DIR}|seed-4_name-Hopper-v5|1"
    "trpo_walker2d_seed5.json|${TRPO_LOG_DIR}|seed-5_name-Walker2d-v5|1"
    "trpo_ant_seed0.json|${TRPO_LOG_DIR}|seed-0_name-Ant-v5|1"
    "trpo_humanoid_seed2.json|${TRPO_LOG_DIR}|seed-2_name-Humanoid-v5|1"
    # SAC / MuJoCo
    "sac_ant_seed1.json|${SAC_LOG_DIR}|seed-1_name-Ant-v5|0"
    "sac_hopper_seed4.json|${SAC_LOG_DIR}|seed-4_name-Hopper-v5|0"
    "sac_halfcheetah_seed5.json|${SAC_LOG_DIR}|seed-5_name-HalfCheetah-v5|0"
    "sac_walker2d_seed5.json|${SAC_LOG_DIR}|seed-5_name-Walker2d-v5|0"
    "sac_humanoid_seed8.json|${SAC_LOG_DIR}|seed-8_name-Humanoid-v5|0"
    # TD3 / MuJoCo
    "td3_ant_seed5.json|${TD3_LOG_DIR}|seed-5_name-Ant-v5|0"
    "td3_hopper_seed6.json|${TD3_LOG_DIR}|seed-6_name-Hopper-v5|0"
    "td3_halfcheetah_seed5.json|${TD3_LOG_DIR}|seed-5_name-HalfCheetah-v5|0"
    "td3_walker2d_seed5.json|${TD3_LOG_DIR}|seed-5_name-Walker2d-v5|0"
    "td3_humanoid_seed9.json|${TD3_LOG_DIR}|seed-9_name-Humanoid-v5|0"
    # DQN / Atari
    "dqn_breakout_seed5.json|${DQN_LOG_DIR}|seed-5_name-BreakoutNoFrameskip-v4_nstep-1|0"
    "dqn_seaquest_seed8.json|${DQN_LOG_DIR}|seed-8_name-SeaquestNoFrameskip-v4_nstep-1|0"
    "dqn_qbert_seed7.json|${DQN_LOG_DIR}|seed-7_name-QbertNoFrameskip-v4_nstep-1|0"
    "dqn_spaceinvaders_seed3.json|${DQN_LOG_DIR}|seed-3_name-SpaceInvadersNoFrameskip-v4_nstep-1|0"
    "dqn_beamrider_seed5.json|${DQN_LOG_DIR}|seed-5_name-BeamRiderNoFrameskip-v4_nstep-1|0"
    "dqn_freeway_seed7.json|${DQN_LOG_DIR}|seed-7_name-FreewayNoFrameskip-v4_nstep-1|0"
    "dqn_pong_seed0.json|${DQN_LOG_DIR}|seed-0_name-PongNoFrameskip-v4_nstep-3|0"
    # PPO / Atari
    "ppo_atari_breakout_seed3.json|${PPO_ATARI_LOG_DIR}|seed-3_name-BreakoutNoFrameskip-v4|0"
    "ppo_atari_seaquest_seed5.json|${PPO_ATARI_LOG_DIR}|seed-5_name-SeaquestNoFrameskip-v4|0"
    "ppo_atari_qbert_seed9.json|${PPO_ATARI_LOG_DIR}|seed-9_name-QbertNoFrameskip-v4|0"
    "ppo_atari_spaceinvaders_seed1.json|${PPO_ATARI_LOG_DIR}|seed-1_name-SpaceInvadersNoFrameskip-v4|0"
    "ppo_atari_beamrider_seed6.json|${PPO_ATARI_LOG_DIR}|seed-6_name-BeamRiderNoFrameskip-v4|0"
    "ppo_atari_pong_seed8.json|${PPO_ATARI_LOG_DIR}|seed-8_name-PongNoFrameskip-v4|0"
    "ppo_atari_freeway_seed1.json|${PPO_ATARI_LOG_DIR}|seed-1_name-FreewayNoFrameskip-v4|0"
)

run_lane() {
    local lane_index="$1"
    local gpu_id="$2"
    shift 2
    local task_index config_file log_dir run_name expect_obs_rms

    for ((task_index=lane_index; task_index<${#TASKS[@]}; task_index+=${#GPUS[@]})); do
        IFS='|' read -r config_file log_dir run_name expect_obs_rms <<< "${TASKS[${task_index}]}"
        echo "[lane ${lane_index}] gpu=${gpu_id} task=$((task_index + 1))/${#TASKS[@]} ${run_name}"
        run_job "${CONFIG_DIR}/${config_file}" "${log_dir}" "${run_name}" "${gpu_id}" "${expect_obs_rms}" "$@"
    done
}

PIDS=()
for lane_index in "${!GPUS[@]}"; do
    run_lane "${lane_index}" "${GPUS[${lane_index}]}" "$@" &
    PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        STATUS=1
    fi
done

if (( STATUS != 0 )); then
    echo "ERROR: one or more GPU lanes failed" >&2
    exit "${STATUS}"
fi

if [[ "${DRY_RUN}" == true ]]; then
    echo "Dry run completed for all ${#TASKS[@]} best-seed jobs. RUN_TAG=${RUN_TAG}"
else
    echo "All ${#TASKS[@]} best-seed jobs completed successfully. RUN_TAG=${RUN_TAG}"
fi
