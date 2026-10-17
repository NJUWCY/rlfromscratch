#!/usr/bin/env bash
set -euo pipefail
# Baseline: round4 2026-10-13_09-56-59, best last-three-evaluation mean (seed 0).
# Fixed temperature/actor settings and discriminator grid: round5_discriminator.json.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" round5_discriminator.json "$@"
