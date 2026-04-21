#!/usr/bin/env bash
set -uo pipefail

cd /root/workspace/yolodist || exit 1
source /root/workspace/.venv/bin/activate
export PYTHONPATH=src

mkdir -p runs/overnight_logs
log_file="runs/overnight_logs/epfa_tuned_remaining.log"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$log_file"
}

run_step() {
  local label="$1"
  shift
  log "START ${label}"
  if "$@" >>"$log_file" 2>&1; then
    log "DONE ${label}"
    return 0
  fi
  log "RETRY ${label}"
  if "$@" >>"$log_file" 2>&1; then
    log "DONE ${label} (retry)"
    return 0
  fi
  log "FAIL ${label}"
  return 1
}

while pgrep -af 'distillation_deeppcb_epfa_tuned.toml' >/dev/null 2>&1; do
  sleep 60
done

run_step "PKU distill_epfa_tuned train" python experiments/distillation/train.py --config configs/train/distillation_pku_market_pcb_epfa_tuned.toml || true
run_step "PKU distill_epfa_tuned eval" python experiments/distillation/evaluate.py --config configs/eval/distillation_pku_market_pcb_epfa_tuned.toml || true

run_step "DsPCBSD+ distill_epfa_tuned train" python experiments/distillation/train.py --config configs/train/distillation_dspcbsd_plus_epfa_tuned.toml || true
run_step "DsPCBSD+ distill_epfa_tuned eval" python experiments/distillation/evaluate.py --config configs/eval/distillation_dspcbsd_plus_epfa_tuned.toml || true

run_step "summarize" python tools/summarize_run.py --pcb-table-root runs --output-dir runs/paper_tables || true

log "PIPELINE COMPLETE epfa_tuned_remaining"
