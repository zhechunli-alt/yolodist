#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/deeppcb_yolox_nano_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src:$ROOT/external/YOLOX"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

if [[ ! -d "$ROOT/external/YOLOX" ]]; then
  log "ERROR external/YOLOX is missing. Clone YOLOX before running this pipeline."
  exit 1
fi

log "START prepare DeepPCB COCO"
python "$ROOT/tools/prepare_deeppcb_coco.py" 2>&1 | tee -a "$MASTER_LOG"
log "DONE  prepare DeepPCB COCO"

log "START install YOLOX editable"
PIP_NO_BUILD_ISOLATION=1 pip install -e "$ROOT/external/YOLOX" 2>&1 | tee -a "$MASTER_LOG"
log "DONE  install YOLOX editable"

log "START YOLOX-Nano train"
python "$ROOT/external/YOLOX/tools/train.py" \
  -f "$ROOT/configs/models/yolox_deeppcb_nano.py" \
  -d 1 \
  -b 16 \
  --fp16 \
  -o \
  --logger tensorboard \
  2>&1 | tee -a "$MASTER_LOG"
log "DONE  YOLOX-Nano train"

log "START YOLOX-Nano eval"
python "$ROOT/external/YOLOX/tools/eval.py" \
  -f "$ROOT/configs/models/yolox_deeppcb_nano.py" \
  -c "$ROOT/runs/deeppcb_compare_yolox/yolox_deeppcb_nano/best_ckpt.pth" \
  -b 16 \
  -d 1 \
  --conf 0.001 \
  --fp16 \
  2>&1 | tee -a "$MASTER_LOG"
log "DONE  YOLOX-Nano eval"
