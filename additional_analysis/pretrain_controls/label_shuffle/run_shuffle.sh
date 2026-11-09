#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SHUFFLE_PYTHON:-python}"
TASK="${1:-missense-CaLM}"
GPU="${2:-0}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/${TASK}_${STAMP}.log"
# Verify the actual GPU/data/model path before launching the full experiment.
"$PYTHON" -u "$SCRIPT_DIR/Pretrain_shuffle.py" --task "$TASK" --gpu "$GPU" --smoke-test > "$LOG_DIR/${TASK}_${STAMP}_smoke.log" 2>&1
nohup setsid "$PYTHON" -u "$SCRIPT_DIR/Pretrain_shuffle.py" --task "$TASK" --gpu "$GPU" --epochs 16 > "$LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$LOG_DIR/${TASK}_${STAMP}.pid"
printf 'PID=%s\nLOG=%s\n' "$TRAIN_PID" "$LOG"
