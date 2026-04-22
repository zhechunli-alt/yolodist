#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/yolodist"
LOG="$ROOT/runs/overnight_logs/finalize_fcos_then_publish.log"

cd "$ROOT"
mkdir -p runs/overnight_logs
source /root/workspace/.venv/bin/activate

log() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"
}

while pgrep -af 'comparison/train.py --config configs/train/comparison_deeppcb_fcos.toml' >/dev/null; do
  sleep 60
done

if [ -f runs/deeppcb_compare/fcos_r50_fpn/weights/best.pt ] && [ ! -f runs/deeppcb_compare/fcos_r50_fpn_eval2/metrics_summary.json ]; then
  log "START fcos eval"
  python experiments/comparison/evaluate.py --config configs/eval/comparison_deeppcb_fcos.toml >> "$LOG" 2>&1
fi

if [ -f runs/deeppcb_compare/fcos_r50_fpn/weights/best.pt ]; then
  log "START regenerate single-model diagnostics"
  python tools/generate_deeppcb_single_model_diagnostics.py >> "$LOG" 2>&1
fi

if [ -f runs/paper_figures/generated/deeppcb_retinanet_r50_fpn_single_pred.png ] && [ -f runs/paper_figures/generated/deeppcb_fcos_r50_fpn_single_pred.png ]; then
  log "START git finalize"
  git add src/yolodist/compare/runner.py configs/train configs/eval docs runs/paper_figures/FIGURE_GUIDE.md runs/paper_tables/deeppcb_external_model_pool.* tools/*.py tools/*.sh || true
  git commit -m 'finish DeepPCB comparison diagnostics and recovery' >> "$LOG" 2>&1 || true
  git push origin codex/ppla-ready >> "$LOG" 2>&1 || true
  log "START rsync retries"
  ./tools/rsync_assets_to_mac.sh >> "$LOG" 2>&1 || true
fi

log "DONE fcos finalize pipeline"
