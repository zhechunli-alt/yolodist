#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/deeppcb_comparison_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src"

run_step() {
  local name="$1"
  shift
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $name" | tee -a "$MASTER_LOG"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE  $name" | tee -a "$MASTER_LOG"
}

run_step "DeepPCB YOLOv8n train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_yolov8n.toml"
run_step "DeepPCB YOLOv8n eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_yolov8n.toml"
run_step "DeepPCB YOLOv10n train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_yolov10n.toml"
run_step "DeepPCB YOLOv10n eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_yolov10n.toml"
run_step "DeepPCB SSDLite train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_ssdlite320.toml"
run_step "DeepPCB SSDLite eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_ssdlite320.toml"
run_step "DeepPCB RetinaNet train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_retinanet.toml"
run_step "DeepPCB RetinaNet eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_retinanet.toml"
run_step "DeepPCB FCOS train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_fcos.toml"
run_step "DeepPCB FCOS eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_fcos.toml"
