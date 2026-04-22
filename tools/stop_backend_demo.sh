#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/yolodist"
PID_FILE="$ROOT/runs/service_logs/backend_demo.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "backend demo pid file not found"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "backend demo stopped: pid=$PID"
else
  echo "backend demo process already exited: pid=$PID"
fi

rm -f "$PID_FILE"
