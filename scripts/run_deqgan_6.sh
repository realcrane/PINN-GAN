#!/usr/bin/env bash

set -uo pipefail

DEVICE="${1:-cuda:0}"
PYTHON_BIN="${PYTHON:-python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs/deqgan"
mkdir -p "$LOG_DIR"

overall_exit=0
for pkey in pos lap1 lap2 kle rea burv; do
    echo "START $pkey on $DEVICE"
    if (
        cd "$REPO_DIR" || exit 1
        MPLBACKEND=Agg "$PYTHON_BIN" -u run_experiment.py \
            --trainer deqgan --pkey "$pkey" --device "$DEVICE"
    ) > "$LOG_DIR/${pkey}.log" 2>&1; then
        echo "DONE  $pkey"
    else
        echo "FAIL  $pkey; see $LOG_DIR/${pkey}.log" >&2
        overall_exit=1
    fi
done

exit "$overall_exit"
