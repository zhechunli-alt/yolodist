#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/watch_remaining_pcb_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

wait_for_training() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    log "WAIT  $pattern"
    sleep 60
  done
  log "DONE  wait $pattern"
}

run_if_missing() {
  local name="$1"
  local done_file="$2"
  shift 2
  if [[ -f "$done_file" ]]; then
    log "SKIP  $name (found $done_file)"
    return 0
  fi
  log "START $name"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  log "DONE  $name"
}

log "Watcher started for remaining PCB locfg jobs"

wait_for_training "experiments/distillation/train.py --config .*distillation_dspcbsd_plus_plain_locfg.toml"
wait_for_training "experiments/distillation/train.py --config .*distillation_dspcbsd_plus_epfa_locfg.toml"

run_if_missing \
  "DsPCBSD+ distill_plain_locfg test-eval" \
  "$ROOT/runs/dspcbsd_plus_distill_locfg/distill_plain_locfg_eval2/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_plain_locfg.toml"

run_if_missing \
  "DsPCBSD+ distill_epfa_locfg test-eval" \
  "$ROOT/runs/dspcbsd_plus_distill_epfa_locfg/distill_epfa_locfg_eval2/metrics_summary.json" \
  python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_epfa_locfg.toml"

run_if_missing \
  "Summarize PCB tables" \
  "$ROOT/runs/paper_tables/pcb_main_results.csv" \
  python "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables"

log "Watcher finished for remaining PCB locfg jobs"
