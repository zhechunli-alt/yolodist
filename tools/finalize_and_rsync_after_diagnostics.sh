#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/yolodist"
LOG="$ROOT/runs/overnight_logs/finalize_and_rsync_after_diagnostics.log"

cd "$ROOT"
mkdir -p runs/overnight_logs

log() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"
}

have_all_diag() {
  [ -f runs/paper_figures/generated/deeppcb_retinanet_r50_fpn_single_pred.png ] && \
  [ -f runs/paper_figures/generated/deeppcb_fcos_r50_fpn_single_pred.png ] && \
  [ -f runs/paper_figures/generated/deeppcb_retinanet_r50_fpn_pr_curve.png ] && \
  [ -f runs/paper_figures/generated/deeppcb_fcos_r50_fpn_pr_curve.png ] && \
  [ -f runs/paper_figures/generated/deeppcb_retinanet_r50_fpn_confusion_matrix.png ] && \
  [ -f runs/paper_figures/generated/deeppcb_fcos_r50_fpn_confusion_matrix.png ]
}

while ! have_all_diag; do
  sleep 60
done

log "All diagnostics ready, start git finalize"
git add src/yolodist/compare/runner.py configs/train configs/eval docs runs/paper_figures/FIGURE_GUIDE.md runs/paper_tables/deeppcb_external_model_pool.* tools/*.py tools/*.sh || true
git commit -m 'add comparison diagnostics, checkpoint recovery, and sync automation' >> "$LOG" 2>&1 || true
git push origin codex/ppla-ready >> "$LOG" 2>&1 || true
log "Git finalize done, start rsync retries"
./tools/rsync_assets_to_mac.sh >> "$LOG" 2>&1 || true
log "Finalize pipeline done"
