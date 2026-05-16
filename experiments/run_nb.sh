#!/usr/bin/env bash
# Usage: run_nb.sh <notebook.ipynb> [log_name]
# Executes a notebook in-place (writes _out.ipynb) via nohup.
# No nbconvert timeout — relies entirely on the notebook's internal TIMEOUT_SEC.
#
# Examples:
#   ./experiments/run_nb.sh experiments/notebooks/test/2026-05-16/test_foo.ipynb
#   ./experiments/run_nb.sh experiments/notebooks/test/2026-05-16/test_foo.ipynb foo

set -euo pipefail

NB_PATH=$(realpath "$1")
NB_DIR=$(dirname "$NB_PATH")
NB_BASE=$(basename "$NB_PATH" .ipynb)

if [[ -n "${2-}" ]]; then
    LOG_NAME="$2"
else
    LOG_NAME="$NB_BASE"
fi

LOG="/home/yin/DAG/experiments/logs/${LOG_NAME}.log"
OUT="${NB_DIR}/${NB_BASE}_out.ipynb"

JUPYTER="/home/yin/DAG/.venv/bin/jupyter"

echo "Launching: $NB_BASE"
echo "  output : $OUT"
echo "  log    : $LOG"

nohup "$JUPYTER" nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    "$NB_PATH" \
    --output "$OUT" \
    > "$LOG" 2>&1 &

echo "  PID    : $!"
