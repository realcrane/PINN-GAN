#!/usr/bin/env bash

set -uo pipefail

DEVICE="${1:-cuda:0}"
PYTHON_BIN="${PYTHON:-python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs/ntk"
mkdir -p "$LOG_DIR"

overall_exit=0
for problem in pos lap1 lap2 wav hea burv; do
    for method in gan lsgan wgan; do
        pkey="${problem}_ntk_${method}"
        echo "START $pkey on $DEVICE"
        if (
            cd "$REPO_DIR" || exit 1
            MPLBACKEND=Agg "$PYTHON_BIN" -u run_experiment.py \
                --trainer ntk --pkey "$pkey" --device "$DEVICE"
        ) > "$LOG_DIR/${pkey}.log" 2>&1; then
            echo "DONE  $pkey"
        else
            echo "FAIL  $pkey; see $LOG_DIR/${pkey}.log" >&2
            overall_exit=1
        fi
    done
done

exit "$overall_exit"
