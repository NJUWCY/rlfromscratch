#!/usr/bin/env bash
set -eo pipefail

# Default: critic search -> stability search -> three-seed confirmation.
# Each stage finishes before the next starts. The last three evaluations select
# the candidate whose hyperparameters (not checkpoint weights) are inherited.
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
stage=all
from_run=""
log_dir=""
dry_run=false
forward=()
grid_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            [[ $# -ge 2 ]] || { echo '--stage requires a value' >&2; exit 2; }
            stage="$2"; shift 2 ;;
        --stage=*) stage="${1#*=}"; shift ;;
        --from-run)
            [[ $# -ge 2 ]] || { echo '--from-run requires a directory' >&2; exit 2; }
            from_run="$2"; shift 2 ;;
        --from-run=*) from_run="${1#*=}"; shift ;;
        --log-dir)
            [[ $# -ge 2 ]] || { echo '--log-dir requires a directory' >&2; exit 2; }
            log_dir="$2"; shift 2 ;;
        --log-dir=*) log_dir="${1#*=}"; shift ;;
        --dry-run) dry_run=true; shift ;;
        --limit|--start|--seed)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            grid_args+=("$1" "$2"); shift 2 ;;
        --limit=*|--start=*|--seed=*|--shuffle)
            grid_args+=("$1"); shift ;;
        --) shift ;;
        log_dir=*|hydra.run.dir=*)
            echo 'Use --log-dir; per-run Hydra/log directories are managed by the launcher.' >&2
            exit 2 ;;
        --help|-h)
            cat <<'HELP'
Usage: bash search/HyPESAC/halfcheetah_param_search.sh [options] [Hydra overrides]
  --stage all          Run critic -> stability -> confirm automatically (default)
  --stage critic       Run only the 12 optimizer / learning-rate / target-tau combinations
  --stage stability    16 update-ratio / discriminator / expert-mixing combinations
  --stage confirm      3 seeds, 1000000 policy steps; requires --from-run
  --from-run DIR       Inherit hyperparameters from a previous run's .hydra/config.yaml
  --dry-run            Preview commands and dependent stages without launching training
  --workers N --gpus 0,1 --limit N --shuffle --seed N --start N
  --log-dir DIR --timeout-hours HOURS

Default concurrency: WORKERS=8, GPUS=0,0,0,0,1,1,1,1 (4 jobs per GPU). No cloud logger or API key is needed.
In all mode, --log-dir is the pipeline root; each stage gets its own subdirectory.
The winner is the completed finite run with the highest mean of its last 3 evaluations.
--limit/--start/--shuffle/--seed affect only the first two grids in all mode;
confirmation still runs all 3 seeds. Hydra key=value overrides affect every stage.
--from-run is for manual stability/confirm stages; all mode selects candidates itself.
Later Hydra overrides take priority. Changing a searched key can collapse the grid.
Example: --stage stability --from-run /absolute/path/to/critic-trial --dry-run
HELP
            exit 0 ;;
        *) forward+=("$1"); shift ;;
    esac
done
case "$stage" in all|critic|stability|confirm) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
if [[ ( "$stage" == all || "$stage" == critic ) && -n "$from_run" ]]; then
    echo "Use --from-run for the stability or confirm stage, not the critic grid." >&2
    exit 2
fi
if [[ "$stage" == confirm && -z "$from_run" ]]; then
    echo 'Confirmation requires --from-run pointing to the candidate to validate.' >&2
    exit 2
fi
if [[ "$stage" == all ]]; then
    for arg in "${forward[@]}"; do
        case "$arg" in
            seed=*|+seed=*|++seed=*)
                echo 'All mode keeps confirmation seeds 0,1,2; use a manual --stage to override seed.' >&2
                exit 2 ;;
        esac
    done
fi

CONDA_BASE="${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-rlzero}"
set -u
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-60}"

SEARCH_DIR="$REPO_DIR/search/HyPESAC"
SEARCH_TOOLS="$SEARCH_DIR/halfcheetah_search_tools.py"
if [[ -z "$log_dir" ]]; then
    label="$stage"
    [[ "$stage" != all ]] || label=pipeline
    log_dir="$REPO_DIR/outputs/HyPESAC-HalfCheetah-${label}-search/$(date +%Y-%m-%d_%H-%M-%S)-$$"
fi
log_dir="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$log_dir")"

run_stage() {
    local current_stage="$1" source_run="$2" run_dir="$3"
    local preset="$SEARCH_DIR/halfcheetah_${current_stage}_search.json"
    local config="$run_dir/search_spec.json" status=0
    local -a selection=() filters=() preview=()
    # Never mix new trials with an older sweep, which could change the winner.
    if [[ -e "$run_dir" ]]; then
        echo "Stage directory already exists; use a new --log-dir: $run_dir" >&2
        return 2
    fi
    mkdir -p "$run_dir"
    if [[ -n "$source_run" ]]; then
        python "$SEARCH_TOOLS" inherit --spec "$preset" \
            --from-run "$source_run" --output "$config"
    else
        cp "$preset" "$config"
        if [[ "$current_stage" == stability ]]; then
            echo 'No --from-run: using provisional Adam / lr=3e-4 / tau=0.005 defaults.'
        fi
    fi
    if [[ "$stage" != all || "$current_stage" != confirm ]]; then
        filters=("${grid_args[@]}")
    fi
    if [[ "$dry_run" == true ]]; then
        preview=(--dry-run)
    fi
    printf '\n[HyPE search] stage=%s output=%s\n' "$current_stage" "$run_dir"
    if python "$REPO_DIR/parallel_search.py" \
        --config "$config" \
        --workers "${WORKERS:-8}" \
        --gpus "${GPUS:-0,0,0,0,1,1,1,1}" \
        --timeout-hours "${TIMEOUT_HOURS:-24}" \
        "${forward[@]}" "${filters[@]}" "${preview[@]}" --log-dir "$run_dir"; then
        status=0
    else
        status=$?
        # Interrupts stop the pipeline. Ordinary failed trials can be excluded
        # from selection if the search stage still has valid completed trials.
        if [[ "$status" -ge 128 ]]; then
            return "$status"
        fi
        echo "Stage $current_stage reported exit $status; checking completed trials." >&2
    fi
    if [[ "$dry_run" == true ]]; then
        return "$status"
    fi
    if [[ "$current_stage" == confirm ]]; then
        selection=(--require-complete)
    else
        selection=(--best-output "$run_dir/best.json")
    fi
    python "$SEARCH_TOOLS" summarize "$run_dir" "${selection[@]}"
    if [[ "$current_stage" == confirm && "$status" -ne 0 ]]; then
        return "$status"
    fi
}

selected_run() {
    python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "$1/best.json"
}

if [[ "$stage" == all ]]; then
    echo "Pipeline output: $log_dir"
    run_stage critic "" "$log_dir/01_critic"
    if [[ "$dry_run" == true ]]; then
        echo 'Next: select the best completed critic candidate by its last 3 evaluations.'
        echo "Then: 16 stability combinations inherit that candidate -> $log_dir/02_stability"
        echo 'Then: select the best completed stability candidate using the same rule.'
        echo "Finally: seeds 0,1,2 inherit it, default 1000000 policy steps each -> $log_dir/03_confirm"
        echo 'Dependent-stage commands are generated after real results exist; no training was started.'
        exit 0
    fi
    critic_winner="$(selected_run "$log_dir/01_critic")"
    echo "Selected critic candidate: $critic_winner"
    run_stage stability "$critic_winner" "$log_dir/02_stability"
    stability_winner="$(selected_run "$log_dir/02_stability")"
    echo "Selected stability candidate: $stability_winner"
    run_stage confirm "$stability_winner" "$log_dir/03_confirm"
    echo "All three stages completed. Confirmation results: $log_dir/03_confirm/summary.csv"
else
    run_stage "$stage" "$from_run" "$log_dir"
fi
