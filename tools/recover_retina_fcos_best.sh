#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/yolodist"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"
source /root/workspace/.venv/bin/activate

echo "[`date -u +%FT%TZ`] START retinanet retrain"
python experiments/comparison/train.py --config configs/train/comparison_deeppcb_retinanet.toml
echo "[`date -u +%FT%TZ`] START retinanet eval"
python experiments/comparison/evaluate.py --config configs/eval/comparison_deeppcb_retinanet.toml

echo "[`date -u +%FT%TZ`] START fcos retrain"
python experiments/comparison/train.py --config configs/train/comparison_deeppcb_fcos.toml
echo "[`date -u +%FT%TZ`] START fcos eval"
python experiments/comparison/evaluate.py --config configs/eval/comparison_deeppcb_fcos.toml

echo "[`date -u +%FT%TZ`] DONE retina/fcos checkpoint recovery"
