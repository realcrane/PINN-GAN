#!/usr/bin/env bash

set -uo pipefail

DEVICE="${1:-cuda:0}"
PYTHON_BIN="${PYTHON:-python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs/ntk_norb"
mkdir -p "$LOG_DIR"

overall_exit=0
for problem in pos lap1 lap2 kle rea burv; do
    for method in gan lsgan wgan; do
        pkey="${problem}_ntk_${method}"
        echo "START $pkey without rollback on $DEVICE"
        if (
            cd "$REPO_DIR" || exit 1
            MPLBACKEND=Agg "$PYTHON_BIN" -u run_experiment.py \
                --trainer ntk --pkey "$pkey" --disable-rollback \
                --device "$DEVICE"
        ) > "$LOG_DIR/${pkey}.log" 2>&1; then
            echo "DONE  $pkey without rollback"
        else
            echo "FAIL  $pkey without rollback; see $LOG_DIR/${pkey}.log" >&2
            overall_exit=1
        fi
    done
done

exit "$overall_exit"
