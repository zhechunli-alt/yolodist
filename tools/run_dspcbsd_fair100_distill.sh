#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/dspcbsd_fair100_$(date -u +%Y%m%dT%H%M%SZ).log"

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

run_step \
  "DsPCBSD+ baseline test-eval" \
  python "$ROOT/experiments/baseline/evaluate.py" --config "$ROOT/configs/eval/baseline_dspcbsd_plus.toml"

run_step \
  "DsPCBSD+ distill_plain_locfg_fair100 train" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_plain_locfg_fair100.toml"

run_step \
  "DsPCBSD+ distill_plain_locfg_fair100 test-eval" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_plain_locfg_fair100.toml"

run_step \
  "DsPCBSD+ distill_epfa_locfg_fair100 train" \
  python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_epfa_locfg_fair100.toml"

run_step \
  "DsPCBSD+ distill_epfa_locfg_fair100 test-eval" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_epfa_locfg_fair100.toml"

run_step \
  "Summarize PCB tables" \
  python "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables"
