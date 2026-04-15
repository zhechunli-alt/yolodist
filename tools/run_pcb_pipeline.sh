#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET=""
MODE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "--dataset is required" >&2
  exit 1
fi

RAW_DIR="$ROOT/datasets/$DATASET"
PROCESSED_DIR="$ROOT/datasets/processed/${DATASET}_detection"
DATA_CONFIG="$ROOT/configs/data/${DATASET}.toml"

prepare_dataset() {
  "$PYTHON_BIN" "$ROOT/tools/prepare_pcb_detection.py" \
    --dataset "$DATASET" \
    --src "$RAW_DIR" \
    --out "$PROCESSED_DIR"
}

train_baseline() {
  "$PYTHON_BIN" "$ROOT/experiments/baseline/train.py" --config "$ROOT/configs/train/baseline_${DATASET}.toml"
  "$PYTHON_BIN" "$ROOT/experiments/baseline/evaluate.py" --config "$ROOT/configs/eval/baseline_${DATASET}.toml"
  cp -f "$ROOT/runs/${DATASET}_baseline/baseline_plain/weights/best.pt" "$ROOT/weights/teacher_${DATASET}.pt"
}

train_student_plain() {
  "$PYTHON_BIN" "$ROOT/experiments/modified_model/train.py" --config "$ROOT/configs/train/modified_model_${DATASET}_plain.toml"
  "$PYTHON_BIN" "$ROOT/experiments/modified_model/evaluate.py" --config "$ROOT/configs/eval/modified_model_${DATASET}_plain.toml"
}

train_epfa() {
  "$PYTHON_BIN" "$ROOT/experiments/modified_model/train.py" --config "$ROOT/configs/train/modified_model_${DATASET}_epfa.toml"
  "$PYTHON_BIN" "$ROOT/experiments/modified_model/evaluate.py" --config "$ROOT/configs/eval/modified_model_${DATASET}_epfa.toml"
}

train_distill_plain() {
  "$PYTHON_BIN" "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_${DATASET}_plain.toml"
  "$PYTHON_BIN" "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_${DATASET}_plain.toml"
}

train_distill_epfa() {
  "$PYTHON_BIN" "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_${DATASET}_epfa.toml"
  "$PYTHON_BIN" "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_${DATASET}_epfa.toml"
}

prepare_dataset

case "$MODE" in
  baseline)
    train_baseline
    train_student_plain
    ;;
  epfa)
    train_epfa
    ;;
  distill)
    train_distill_plain
    train_distill_epfa
    ;;
  all)
    train_baseline
    train_student_plain
    train_epfa
    train_distill_plain
    train_distill_epfa
    ;;
  *)
    echo "Unsupported --mode: $MODE" >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables"
