#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/distill_locfg_$(date -u +%Y%m%dT%H%M%SZ).log"

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

maybe_run() {
  local name="$1"
  local done_file="$2"
  shift 2
  if [[ -f "$done_file" ]]; then
    log "SKIP  $name (found $done_file)"
    return 0
  fi
  run_step "$name" "$@"
}

copy_teacher_if_needed() {
  local src="$1"
  local dst="$2"
  if [[ -f "$dst" ]]; then
    log "SKIP  teacher copy (found $dst)"
    return 0
  fi
  if [[ ! -f "$src" ]]; then
    log "MISS  teacher source not found: $src"
    return 1
  fi
  cp -f "$src" "$dst"
  log "COPY  $(basename "$src") -> $(basename "$dst")"
}

log "Localization + foreground KD overnight pipeline started"

# Ensure teacher checkpoints exist
copy_teacher_if_needed "$ROOT/runs/deeppcb_baseline/baseline_plain/weights/best.pt" "$ROOT/weights/teacher_deeppcb.pt"
copy_teacher_if_needed "$ROOT/runs/pku_market_pcb_baseline/baseline_plain/weights/best.pt" "$ROOT/weights/teacher_pku_market_pcb.pt"
copy_teacher_if_needed "$ROOT/runs/dspcbsd_plus_baseline/baseline_plain/weights/best.pt" "$ROOT/weights/teacher_dspcbsd_plus.pt"

# DeepPCB
maybe_run \
  "DeepPCB distill_plain_locfg train" \
  "$ROOT/runs/deeppcb_distill_locfg/distill_plain_locfg/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_deeppcb_plain_locfg.toml"
maybe_run \
  "DeepPCB distill_plain_locfg test-eval" \
  "$ROOT/runs/deeppcb_distill_locfg/distill_plain_locfg_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_deeppcb_plain_locfg.toml"
maybe_run \
  "DeepPCB distill_epfa_locfg train" \
  "$ROOT/runs/deeppcb_distill_epfa_locfg/distill_epfa_locfg/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_deeppcb_epfa_locfg.toml"
maybe_run \
  "DeepPCB distill_epfa_locfg test-eval" \
  "$ROOT/runs/deeppcb_distill_epfa_locfg/distill_epfa_locfg_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_deeppcb_epfa_locfg.toml"

# PKU-Market-PCB
maybe_run \
  "PKU distill_plain_locfg train" \
  "$ROOT/runs/pku_market_pcb_distill_locfg/distill_plain_locfg/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_pku_market_pcb_plain_locfg.toml"
maybe_run \
  "PKU distill_plain_locfg test-eval" \
  "$ROOT/runs/pku_market_pcb_distill_locfg/distill_plain_locfg_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_pku_market_pcb_plain_locfg.toml"
maybe_run \
  "PKU distill_epfa_locfg train" \
  "$ROOT/runs/pku_market_pcb_distill_epfa_locfg/distill_epfa_locfg/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_pku_market_pcb_epfa_locfg.toml"
maybe_run \
  "PKU distill_epfa_locfg test-eval" \
  "$ROOT/runs/pku_market_pcb_distill_epfa_locfg/distill_epfa_locfg_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_pku_market_pcb_epfa_locfg.toml"

# DsPCBSD+
maybe_run \
  "DsPCBSD+ distill_plain_locfg train" \
  "$ROOT/runs/dspcbsd_plus_distill_locfg/distill_plain_locfg/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_plain_locfg.toml"
maybe_run \
  "DsPCBSD+ distill_plain_locfg test-eval" \
  "$ROOT/runs/dspcbsd_plus_distill_locfg/distill_plain_locfg_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_plain_locfg.toml"
maybe_run \
  "DsPCBSD+ distill_epfa_locfg train" \
  "$ROOT/runs/dspcbsd_plus_distill_epfa_locfg/distill_epfa_locfg/weights/best.pt" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_epfa_locfg.toml"
maybe_run \
  "DsPCBSD+ distill_epfa_locfg test-eval" \
  "$ROOT/runs/dspcbsd_plus_distill_epfa_locfg/distill_epfa_locfg_eval/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_epfa_locfg.toml"

run_step \
  "Summarize PCB tables" \
  python "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables"

log "Localization + foreground KD overnight pipeline finished"
