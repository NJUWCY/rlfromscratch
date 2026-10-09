#!/usr/bin/env bash
set -euo pipefail
# Only this grid's parameters differ from scripts/hype.sh.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/halfcheetah_search_common.sh" \
    halfcheetah_confirm_search.json "$@"
