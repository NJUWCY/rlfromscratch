#!/usr/bin/env bash
set -eo pipefail
SEARCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SEARCH_DIR/../../.."
spec="${1:?search spec required}"
shift

source /home/ubuntu/wangchenyang/anaconda/etc/profile.d/conda.sh
conda activate rlzero
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

source_run=""
extra_args=()
while (($#)); do
    case "$1" in
        --from-run) source_run="${2:?--from-run requires a run directory}"; shift 2 ;;
        --from-run=*) source_run="${1#*=}"; shift ;;
        *) extra_args+=("$1"); shift ;;
    esac
done
prepared_spec="$(mktemp "${TMPDIR:-/tmp}/hype-humanoid-spec.XXXXXX")"
trap 'rm -f "$prepared_spec"' EXIT
prepare_args=()
if [[ -n "$source_run" ]]; then prepare_args+=(--from-run "$source_run"); fi
python "$SEARCH_DIR/prepare_spec.py" \
    --spec "$SEARCH_DIR/$spec" --output "$prepared_spec" "${prepare_args[@]}"

# Eight concurrent jobs: four on GPU 0 and four on GPU 1.
python parallel_search.py \
    --config "$prepared_spec" \
    --workers "${WORKERS:-8}" --gpus "${GPUS:-0,0,0,0,1,1,1,1}" \
    --timeout-hours "${TIMEOUT_HOURS:-48}" "${extra_args[@]}"
