#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/watch_pku_epfa_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

while pgrep -f "experiments/modified_model/train.py --config .*modified_model_pku_market_pcb_epfa.toml" >/dev/null 2>&1; do
  log "WAIT  pku student_epfa train"
  sleep 60
done

log "DONE  pku student_epfa train"

if [[ ! -f "$ROOT/runs/pku_market_pcb_epfa/student_epfa_eval2/metrics_summary.json" ]]; then
  log "START pku student_epfa test-eval"
  python "$ROOT/experiments/modified_model/evaluate.py" --config "$ROOT/configs/eval/modified_model_pku_market_pcb_epfa.toml" 2>&1 | tee -a "$MASTER_LOG"
  log "DONE  pku student_epfa test-eval"
else
  log "SKIP  pku student_epfa test-eval"
fi

log "START summarize"
python "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables" 2>&1 | tee -a "$MASTER_LOG"
log "DONE  summarize"
