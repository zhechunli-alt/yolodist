#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/watch_dspcbsd_baseline_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

while pgrep -f "experiments/baseline/train.py --config .*baseline_dspcbsd_plus_resume.toml" >/dev/null 2>&1; do
  log "WAIT  dspcbsd_plus baseline resume train"
  sleep 60
done

log "DONE  dspcbsd_plus baseline resume train"

if [[ ! -f "$ROOT/runs/dspcbsd_plus_baseline/baseline_plain_eval2/metrics_summary.json" ]]; then
  log "START dspcbsd_plus baseline test-eval"
  python "$ROOT/experiments/baseline/evaluate.py" --config "$ROOT/configs/eval/baseline_dspcbsd_plus.toml" 2>&1 | tee -a "$MASTER_LOG"
  log "DONE  dspcbsd_plus baseline test-eval"
else
  log "SKIP  dspcbsd_plus baseline test-eval"
fi

log "START summarize"
python "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables" 2>&1 | tee -a "$MASTER_LOG"
log "DONE  summarize"
