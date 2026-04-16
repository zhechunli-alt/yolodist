#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/runs/overnight_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/distill_locfg_parallel_$(date -u +%Y%m%dT%H%M%SZ).log"
MAX_JOBS="${MAX_JOBS:-2}"

if [[ -f /root/workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/workspace/.venv/bin/activate
fi

export PYTHONPATH="$ROOT/src"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$MASTER_LOG"
}

copy_teacher_if_needed() {
  local src="$1"
  local dst="$2"
  if [[ -f "$dst" ]]; then
    log "SKIP  teacher copy (found $dst)"
    return 0
  fi
  cp -f "$src" "$dst"
  log "COPY  $(basename "$src") -> $(basename "$dst")"
}

launch_job() {
  local name="$1"
  local done_file="$2"
  local job_log="$3"
  shift 3
  if [[ -f "$done_file" ]]; then
    log "SKIP  $name (found $done_file)"
    return 0
  fi
  log "START $name"
  (
    "$@" 2>&1 | tee "$job_log"
  ) &
}

wait_for_slot() {
  while true; do
    local running
    running="$(jobs -pr | wc -l)"
    if (( running < MAX_JOBS )); then
      break
    fi
    wait -n || true
  done
}

wait_all() {
  local status=0
  for pid in $(jobs -pr); do
    wait "$pid" || status=$?
  done
  return "$status"
}

run_phase_parallel() {
  local phase_name="$1"
  shift
  log "PHASE $phase_name"
  while (( "$#" > 0 )); do
    local name="$1"; shift
    local done_file="$1"; shift
    local job_log="$1"; shift
    local cmd=()
    while (( "$#" > 0 )) && [[ "$1" != "::JOB::" ]]; do
      cmd+=("$1")
      shift
    done
    if (( "$#" > 0 )) && [[ "$1" == "::JOB::" ]]; then
      shift
    fi
    wait_for_slot
    launch_job "$name" "$done_file" "$job_log" "${cmd[@]}"
  done
  wait_all
  log "DONE  PHASE $phase_name"
}

log "Parallel localization+foreground KD pipeline started"

copy_teacher_if_needed "$ROOT/runs/deeppcb_baseline/baseline_plain/weights/best.pt" "$ROOT/weights/teacher_deeppcb.pt"
copy_teacher_if_needed "$ROOT/runs/pku_market_pcb_baseline/baseline_plain/weights/best.pt" "$ROOT/weights/teacher_pku_market_pcb.pt"
copy_teacher_if_needed "$ROOT/runs/dspcbsd_plus_baseline/baseline_plain/weights/best.pt" "$ROOT/weights/teacher_dspcbsd_plus.pt"

run_phase_parallel "train" \
  "DeepPCB distill_plain_locfg train" "$ROOT/runs/deeppcb_distill_locfg/distill_plain_locfg/weights/best.pt" "$LOG_DIR/deeppcb_distill_plain_locfg.log" \
    python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_deeppcb_plain_locfg.toml" "::JOB::" \
  "DeepPCB distill_epfa_locfg train" "$ROOT/runs/deeppcb_distill_epfa_locfg/distill_epfa_locfg/weights/best.pt" "$LOG_DIR/deeppcb_distill_epfa_locfg.log" \
    python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_deeppcb_epfa_locfg.toml" "::JOB::" \
  "PKU distill_plain_locfg train" "$ROOT/runs/pku_market_pcb_distill_locfg/distill_plain_locfg/weights/best.pt" "$LOG_DIR/pku_distill_plain_locfg.log" \
    python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_pku_market_pcb_plain_locfg.toml" "::JOB::" \
  "PKU distill_epfa_locfg train" "$ROOT/runs/pku_market_pcb_distill_epfa_locfg/distill_epfa_locfg/weights/best.pt" "$LOG_DIR/pku_distill_epfa_locfg.log" \
    python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_pku_market_pcb_epfa_locfg.toml" "::JOB::" \
  "DsPCBSD+ distill_plain_locfg train" "$ROOT/runs/dspcbsd_plus_distill_locfg/distill_plain_locfg/weights/best.pt" "$LOG_DIR/dspcbsd_distill_plain_locfg.log" \
    python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_plain_locfg.toml" "::JOB::" \
  "DsPCBSD+ distill_epfa_locfg train" "$ROOT/runs/dspcbsd_plus_distill_epfa_locfg/distill_epfa_locfg/weights/best.pt" "$LOG_DIR/dspcbsd_distill_epfa_locfg.log" \
    python "$ROOT/experiments/distillation/train.py" --config "$ROOT/configs/train/distillation_dspcbsd_plus_epfa_locfg.toml"

run_phase_parallel "eval" \
  "DeepPCB distill_plain_locfg eval" "$ROOT/runs/deeppcb_distill_locfg/distill_plain_locfg_eval/metrics_summary.json" "$LOG_DIR/deeppcb_distill_plain_locfg_eval.log" \
    python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_deeppcb_plain_locfg.toml" "::JOB::" \
  "DeepPCB distill_epfa_locfg eval" "$ROOT/runs/deeppcb_distill_epfa_locfg/distill_epfa_locfg_eval/metrics_summary.json" "$LOG_DIR/deeppcb_distill_epfa_locfg_eval.log" \
    python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_deeppcb_epfa_locfg.toml" "::JOB::" \
  "PKU distill_plain_locfg eval" "$ROOT/runs/pku_market_pcb_distill_locfg/distill_plain_locfg_eval/metrics_summary.json" "$LOG_DIR/pku_distill_plain_locfg_eval.log" \
    python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_pku_market_pcb_plain_locfg.toml" "::JOB::" \
  "PKU distill_epfa_locfg eval" "$ROOT/runs/pku_market_pcb_distill_epfa_locfg/distill_epfa_locfg_eval/metrics_summary.json" "$LOG_DIR/pku_distill_epfa_locfg_eval.log" \
    python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_pku_market_pcb_epfa_locfg.toml" "::JOB::" \
  "DsPCBSD+ distill_plain_locfg eval" "$ROOT/runs/dspcbsd_plus_distill_locfg/distill_plain_locfg_eval/metrics_summary.json" "$LOG_DIR/dspcbsd_distill_plain_locfg_eval.log" \
    python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_plain_locfg.toml" "::JOB::" \
  "DsPCBSD+ distill_epfa_locfg eval" "$ROOT/runs/dspcbsd_plus_distill_epfa_locfg/distill_epfa_locfg_eval/metrics_summary.json" "$LOG_DIR/dspcbsd_distill_epfa_locfg_eval.log" \
    python "$ROOT/experiments/distillation/evaluate.py" --config "$ROOT/configs/eval/distillation_dspcbsd_plus_epfa_locfg.toml"

log "START summarize"
python "$ROOT/tools/summarize_run.py" --pcb-table-root "$ROOT/runs" --output-dir "$ROOT/runs/paper_tables" 2>&1 | tee -a "$MASTER_LOG"
log "DONE  summarize"
log "Parallel localization+foreground KD pipeline finished"
