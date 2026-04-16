#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/pcb_overnight_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  log "DONE  $name"
}

maybe_run_train_eval() {
  local name="$1"
  local result_file="$2"
  shift 2
  if [[ -f "$result_file" ]]; then
    log "SKIP  $name (found $result_file)"
    return 0
  fi
  run_step "$name" "$@"
}

copy_teacher_if_needed() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$dst" ]]; then
    cp -f "$src" "$dst"
    log "COPY  teacher $(basename "$src") -> $(basename "$dst")"
  else
    log "SKIP  teacher copy (found $dst)"
  fi
}

log "PCB overnight pipeline started"

# DeepPCB alpha=0.5 distillation completion
maybe_run_train_eval \
  "DeepPCB distill_plain alpha0.5 test-eval" \
  "$ROOT/runs/deeppcb_distill_alpha05/distill_plain_alpha05_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_deeppcb_plain_alpha05.toml"

maybe_run_train_eval \
  "DeepPCB distill_epfa alpha0.5 train" \
  "$ROOT/runs/deeppcb_distill_epfa_alpha05/distill_epfa_alpha05/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_deeppcb_epfa_alpha05.toml"

maybe_run_train_eval \
  "DeepPCB distill_epfa alpha0.5 test-eval" \
  "$ROOT/runs/deeppcb_distill_epfa_alpha05/distill_epfa_alpha05_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_deeppcb_epfa_alpha05.toml"

# PKU-Market-PCB full line
maybe_run_train_eval \
  "PKU baseline test-eval" \
  "$ROOT/runs/pku_market_pcb_baseline/baseline_plain_eval/metrics_summary.json" \
  python "$ROOT/experiments/baseline/evaluate.py" --config "$ROOT/configs/eval/baseline_pku_market_pcb.toml"

copy_teacher_if_needed \
  "$ROOT/runs/pku_market_pcb_baseline/baseline_plain/weights/best.pt" \
  "$ROOT/weights/teacher_pku_market_pcb.pt"

maybe_run_train_eval \
  "PKU student_plain test-eval" \
  "$ROOT/runs/pku_market_pcb_modified/student_plain_eval/metrics_summary.json" \
  python "$ROOT/experiments/modified_model/evaluate.py" --config "$ROOT/configs/eval/modified_model_pku_market_pcb_plain.toml"

maybe_run_train_eval \
  "PKU student_epfa train" \
  "$ROOT/runs/pku_market_pcb_epfa/student_epfa/weights/best.pt" \
  python "$ROOT/experiments/modified_model/train.py" --config "$ROOT/configs/train/modified_model_pku_market_pcb_epfa.toml"

maybe_run_train_eval \
  "PKU student_epfa test-eval" \
  "$ROOT/runs/pku_market_pcb_epfa/student_epfa_eval/metrics_summary.json" \
  python "$ROOT/experiments/modified_model/evaluate.py" --config "$ROOT/configs/eval/modified_model_pku_market_pcb_epfa.toml"

maybe_run_train_eval \
  "PKU distill_plain train" \
  "$ROOT/runs/pku_market_pcb_distill/distill_plain/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_pku_market_pcb_plain.toml"

maybe_run_train_eval \
  "PKU distill_plain test-eval" \
  "$ROOT/runs/pku_market_pcb_distill/distill_plain_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_pku_market_pcb_plain.toml"

maybe_run_train_eval \
  "PKU distill_epfa train" \
  "$ROOT/runs/pku_market_pcb_distill_epfa/distill_epfa/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_pku_market_pcb_epfa.toml"

maybe_run_train_eval \
  "PKU distill_epfa test-eval" \
  "$ROOT/runs/pku_market_pcb_distill_epfa/distill_epfa_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_pku_market_pcb_epfa.toml"

# DsPCBSD+ full line
maybe_run_train_eval \
  "DsPCBSD+ baseline train" \
  "$ROOT/runs/dspcbsd_plus_baseline/baseline_plain/weights/best.pt" \
  python "$ROOT/experiments/baseline/train.py" --config "$ROOT/configs/train/baseline_dspcbsd_plus.toml"

maybe_run_train_eval \
  "DsPCBSD+ baseline test-eval" \
  "$ROOT/runs/dspcbsd_plus_baseline/baseline_plain_eval/metrics_summary.json" \
  python "$ROOT/experiments/baseline/evaluate.py" --config "$ROOT/configs/eval/baseline_dspcbsd_plus.toml"

copy_teacher_if_needed \
  "$ROOT/runs/dspcbsd_plus_baseline/baseline_plain/weights/best.pt" \
  "$ROOT/weights/teacher_dspcbsd_plus.pt"

maybe_run_train_eval \
  "DsPCBSD+ distill_plain train" \
  "$ROOT/runs/dspcbsd_plus_distill/distill_plain/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_plain.toml"

maybe_run_train_eval \
  "DsPCBSD+ distill_plain test-eval" \
  "$ROOT/runs/dspcbsd_plus_distill/distill_plain_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_plain.toml"

maybe_run_train_eval \
  "DsPCBSD+ distill_epfa train" \
  "$ROOT/runs/dspcbsd_plus_distill_epfa/distill_epfa/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_epfa.toml"

maybe_run_train_eval \
  "DsPCBSD+ distill_epfa test-eval" \
  "$ROOT/runs/dspcbsd_plus_distill_epfa/distill_epfa_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_epfa.toml"

run_step \
  "Summarize PCB tables" \
  python "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables"

log "PCB overnight pipeline finished"
