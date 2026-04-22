#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/yolodist"
LOG_DIR="$ROOT/runs/service_logs"
LOG_FILE="$LOG_DIR/backend_demo.log"
PID_FILE="$LOG_DIR/backend_demo.pid"

mkdir -p "$LOG_DIR"
cd "$ROOT"

source /root/workspace/.venv/bin/activate

python tools/sync_standard_weights.py >/dev/null || true
python tools/backend_preflight.py

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "backend demo already running: pid=$(cat "$PID_FILE")"
  exit 0
fi

export BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
export BACKEND_PORT="${BACKEND_PORT:-18080}"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

if ss -ltn "( sport = :$BACKEND_PORT )" 2>/dev/null | tail -n +2 | grep -q .; then
  echo "backend demo port already in use: $BACKEND_PORT" >&2
  exit 1
fi

nohup python -m service.backend.app >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "backend demo started"
  echo "host=$BACKEND_HOST port=$BACKEND_PORT pid=$(cat "$PID_FILE")"
  echo "log=$LOG_FILE"
else
  echo "backend demo failed to start; see $LOG_FILE" >&2
  exit 1
fi
