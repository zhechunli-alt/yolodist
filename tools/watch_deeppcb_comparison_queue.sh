#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/deeppcb_comparison_queue_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

has_eval_summary() {
  local pattern="$1"
  compgen -G "$pattern" >/dev/null 2>&1
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  log "DONE  $name"
}

while pgrep -f "experiments/comparison/train.py --config configs/train/comparison_deeppcb_yolov8n.toml" >/dev/null 2>&1; do
  log "WAIT  DeepPCB YOLOv8n train"
  sleep 60
done

if ! has_eval_summary "$ROOT/runs/deeppcb_compare/yolov8n_eval*/metrics_summary.json"; then
  run_step "DeepPCB YOLOv8n eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_yolov8n.toml"
fi

if [[ ! -f "$ROOT/runs/deeppcb_compare/yolov10n/weights/best.pt" ]]; then
  run_step "DeepPCB YOLOv10n train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_yolov10n.toml"
fi
if ! has_eval_summary "$ROOT/runs/deeppcb_compare/yolov10n_eval*/metrics_summary.json"; then
  run_step "DeepPCB YOLOv10n eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_yolov10n.toml"
fi

if [[ ! -f "$ROOT/runs/deeppcb_compare/ssdlite320_mobilenet_v3_large/weights/best.pt" ]]; then
  run_step "DeepPCB SSDLite train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_ssdlite320.toml"
fi
if ! has_eval_summary "$ROOT/runs/deeppcb_compare/ssdlite320_mobilenet_v3_large_eval*/metrics_summary.json"; then
  run_step "DeepPCB SSDLite eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_ssdlite320.toml"
fi

if [[ ! -f "$ROOT/runs/deeppcb_compare/retinanet_r50_fpn/weights/best.pt" ]]; then
  run_step "DeepPCB RetinaNet train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_retinanet.toml"
fi
if ! has_eval_summary "$ROOT/runs/deeppcb_compare/retinanet_r50_fpn_eval*/metrics_summary.json"; then
  run_step "DeepPCB RetinaNet eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_retinanet.toml"
fi

if [[ ! -f "$ROOT/runs/deeppcb_compare/fcos_r50_fpn/weights/best.pt" ]]; then
  run_step "DeepPCB FCOS train" python "$ROOT/experiments/comparison/train.py" --config "$ROOT/configs/train/comparison_deeppcb_fcos.toml"
fi
if ! has_eval_summary "$ROOT/runs/deeppcb_compare/fcos_r50_fpn_eval*/metrics_summary.json"; then
  run_step "DeepPCB FCOS eval" python "$ROOT/experiments/comparison/evaluate.py" --config "$ROOT/configs/eval/comparison_deeppcb_fcos.toml"
fi

if ! has_eval_summary "$ROOT/runs/deeppcb_compare_yolox/yolox_nano_eval*/metrics_summary.json"; then
  run_step "DeepPCB YOLOX-Nano pipeline" bash "$ROOT/tools/run_deeppcb_yolox_nano.sh"
fi
