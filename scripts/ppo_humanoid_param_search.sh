#!/usr/bin/env bash
# Run the PPO Humanoid-v5 parameter sweep in parallel.
#
# This is a thin wrapper around parallel_search.py. The search space, the
# fixed Hydra overrides and the run directory layout are all defined inside
# parallel_search.py. Any extra CLI arguments are forwarded to it, so you can
# do e.g.:
#
#   bash ppo_humanoid_param_search.sh                            
#   bash ppo_humanoid_param_search.sh --workers 4                
#   bash ppo_humanoid_param_search.sh --shuffle --limit 64        
#   bash ppo_humanoid_param_search.sh --gpus 0,1,2,3,4,5,6,7      
#   bash ppo_humanoid_param_search.sh seed=1 total_epoch=1000     
set -e

CONDA_BASE=${CONDA_BASE:-/home/ubuntu/wangchenyang/anaconda}
CONDA_ENV=${CONDA_ENV:-rlzero}

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-yyKpLHGppV78RFW0p1PNQ}

exec python parallel_search.py --workers 16 --gpus 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1  "$@"
