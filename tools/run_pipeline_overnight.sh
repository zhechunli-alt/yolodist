#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/runs/overnight_logs"
mkdir -p "${LOG_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

run_stage() {
  local stage_name="$1"
  shift
  local stage_log="${LOG_DIR}/${stage_name}.log"
  log "START ${stage_name}"
  "$@" 2>&1 | tee "${stage_log}"
  log "DONE ${stage_name}"
}

cd "${ROOT_DIR}"

if [[ -f "/root/workspace/.venv/bin/activate" ]]; then
  # Common server path used in this project.
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
elif [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
else
  log "WARN virtualenv activate script not found, using current python"
fi

run_stage "preflight_before" python tools/preflight_check.py || true
run_stage "prepare_dataset" python tools/prepare_mvtec_detection.py --config configs/data/mvtec_detection.toml
run_stage "train_baseline" python experiments/baseline/train.py --config configs/train/baseline.toml

if [[ ! -f "runs/baseline/yolo11n_baseline/weights/best.pt" ]]; then
  log "ERROR baseline best.pt not found, stopping"
  exit 1
fi

cp -f runs/baseline/yolo11n_baseline/weights/best.pt weights/teacher.pt
log "teacher.pt updated from baseline best.pt"

run_stage "preflight_after_teacher" python tools/preflight_check.py
run_stage "train_modified_ppla" python experiments/modified_model/train.py --config configs/train/modified_model.toml
run_stage "train_distill_ppla" python experiments/distillation/train.py --config configs/train/distillation.toml

run_stage "eval_baseline" python experiments/baseline/evaluate.py --config configs/eval/baseline.toml
run_stage "eval_modified_ppla" python experiments/modified_model/evaluate.py --config configs/eval/modified_model.toml
run_stage "eval_distill_ppla" python experiments/distillation/evaluate.py --config configs/eval/distillation.toml

if [[ -d "runs/baseline/yolo11n_baseline" ]]; then
  run_stage "summary_baseline" python tools/summarize_run.py runs/baseline/yolo11n_baseline
fi
if [[ -d "runs/modified/yolo11_student" ]]; then
  run_stage "summary_modified_ppla" python tools/summarize_run.py runs/modified/yolo11_student
fi
if [[ -d "runs/distill/yolo11_student_distill" ]]; then
  run_stage "summary_distill_ppla" python tools/summarize_run.py runs/distill/yolo11_student_distill
fi

log "ALL DONE"
