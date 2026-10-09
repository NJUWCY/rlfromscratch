#!/usr/bin/env bash
set -euo pipefail
# 3 expert ratios x 2 shared actor/critic optimizers x 3 seeds = 18 runs.
# Reuse hype.sh defaults and the launcher: 8 workers, 4 jobs per GPU (0, 1).
# round1.json pins the previous search network, loss and policy-clock behavior.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../halfcheetah_search_common.sh" \
    further/round1.json "$@"
