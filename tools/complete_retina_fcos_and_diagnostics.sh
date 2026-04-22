#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/yolodist"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"
source /root/workspace/.venv/bin/activate

log() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_DIR/complete_retina_fcos_and_diagnostics.log"
}

wait_for_retina_train() {
  while pgrep -af 'comparison/train.py --config configs/train/comparison_deeppcb_retinanet.toml' >/dev/null; do
    sleep 30
  done
}

log "WAIT retinanet train"
wait_for_retina_train

if [ -f "$ROOT/runs/deeppcb_compare/retinanet_r50_fpn/weights/best.pt" ] && [ ! -f "$ROOT/runs/deeppcb_compare/retinanet_r50_fpn_eval2/metrics_summary.json" ]; then
  log "START retinanet eval"
  python experiments/comparison/evaluate.py --config configs/eval/comparison_deeppcb_retinanet.toml >> "$LOG_DIR/complete_retina_fcos_and_diagnostics.log" 2>&1
fi

if [ ! -f "$ROOT/runs/deeppcb_compare/fcos_r50_fpn/weights/best.pt" ]; then
  log "START fcos train"
  python experiments/comparison/train.py --config configs/train/comparison_deeppcb_fcos.toml >> "$LOG_DIR/complete_retina_fcos_and_diagnostics.log" 2>&1
fi

if [ -f "$ROOT/runs/deeppcb_compare/fcos_r50_fpn/weights/best.pt" ] && [ ! -f "$ROOT/runs/deeppcb_compare/fcos_r50_fpn_eval2/metrics_summary.json" ]; then
  log "START fcos eval"
  python experiments/comparison/evaluate.py --config configs/eval/comparison_deeppcb_fcos.toml >> "$LOG_DIR/complete_retina_fcos_and_diagnostics.log" 2>&1
fi

log "START regenerate single-model diagnostics"
python tools/generate_deeppcb_single_model_diagnostics.py >> "$LOG_DIR/complete_retina_fcos_and_diagnostics.log" 2>&1

log "DONE retina/fcos recovery and diagnostics"
