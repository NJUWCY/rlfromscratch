#!/usr/bin/env bash
set -euo pipefail
SEARCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage=all
if [[ "${1:-}" == --stage ]]; then stage="${2:?--stage requires a value}"; shift 2; fi
case "$stage" in
    all) stages=(critic mixing actor temperature discriminator) ;;
    critic|mixing|actor|temperature|discriminator|confirm) stages=("$stage") ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
# Each independent grid starts from hype.sh; no winner is inherited.
for arg in "$@"; do
    if [[ "$stage" == all && ( "$arg" == --log-dir || "$arg" == --log-dir=* ) ]]; then
        echo 'Use --log-dir with a single --stage or an individual search script.' >&2
        exit 2
    fi
done
for stage in "${stages[@]}"; do
    bash "$SEARCH_DIR/halfcheetah_${stage}_search.sh" "$@"
done
