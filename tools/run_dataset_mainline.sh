#!/usr/bin/env bash
set -uo pipefail

dataset="${1:-}"
mode="${2:-full}"
if [[ -z "$dataset" ]]; then
  echo "usage: $0 <pku_market_pcb|dspcbsd_plus>" >&2
  exit 1
fi

cd /root/workspace/yolodist || exit 1
source /root/workspace/.venv/bin/activate
export PYTHONPATH=src

mkdir -p runs
log_file="runs/${dataset}_mainline.log"

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

run_step_if_missing() {
  local done_pattern="$1"
  shift
  local label="$1"
  shift
  if compgen -G "$done_pattern" >/dev/null; then
    log "SKIP ${label} (found ${done_pattern})"
    return 0
  fi
  run_step "$label" "$@"
}

copy_teacher() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
    log "DONE copy teacher -> ${dst}"
    return 0
  fi
  log "FAIL missing teacher source ${src}"
  return 1
}

case "$dataset" in
  pku_market_pcb)
    baseline_train="configs/train/baseline_pku_market_pcb.toml"
    baseline_eval="configs/eval/baseline_pku_market_pcb.toml"
    plain_train="configs/train/modified_model_pku_market_pcb_plain.toml"
    plain_eval="configs/eval/modified_model_pku_market_pcb_plain.toml"
    epfa_train="configs/train/modified_model_pku_market_pcb_epfa.toml"
    epfa_eval="configs/eval/modified_model_pku_market_pcb_epfa.toml"
    distill_train="configs/train/distillation_pku_market_pcb_epfa.toml"
    distill_eval="configs/eval/distillation_pku_market_pcb_epfa.toml"
    teacher_src="runs/pku_market_pcb_baseline/baseline_plain/weights/best.pt"
    teacher_dst="weights/teacher_pku_market_pcb.pt"
    baseline_train_done="runs/pku_market_pcb_baseline/baseline_plain/weights/best.pt"
    baseline_eval_done="runs/pku_market_pcb_baseline/baseline_plain_eval*/metrics_summary.json"
    plain_train_done="runs/pku_market_pcb_modified/student_plain/weights/best.pt"
    plain_eval_done="runs/pku_market_pcb_modified/student_plain_eval*/metrics_summary.json"
    epfa_train_done="runs/pku_market_pcb_epfa/student_epfa/weights/best.pt"
    epfa_eval_done="runs/pku_market_pcb_epfa/student_epfa_eval*/metrics_summary.json"
    distill_train_done="runs/pku_market_pcb_distill_epfa/distill_epfa_fair/weights/best.pt"
    distill_eval_done="runs/pku_market_pcb_distill_epfa/distill_epfa_fair_eval*/metrics_summary.json"
    ;;
  dspcbsd_plus)
    baseline_train="configs/train/baseline_dspcbsd_plus.toml"
    baseline_eval="configs/eval/baseline_dspcbsd_plus.toml"
    plain_train="configs/train/modified_model_dspcbsd_plus_plain.toml"
    plain_eval="configs/eval/modified_model_dspcbsd_plus_plain.toml"
    epfa_train="configs/train/modified_model_dspcbsd_plus_epfa.toml"
    epfa_eval="configs/eval/modified_model_dspcbsd_plus_epfa.toml"
    distill_train="configs/train/distillation_dspcbsd_plus_epfa.toml"
    distill_eval="configs/eval/distillation_dspcbsd_plus_epfa.toml"
    teacher_src="runs/dspcbsd_plus_baseline/baseline_plain/weights/best.pt"
    teacher_dst="weights/teacher_dspcbsd_plus.pt"
    baseline_train_done="runs/dspcbsd_plus_baseline/baseline_plain/weights/best.pt"
    baseline_eval_done="runs/dspcbsd_plus_baseline/baseline_plain_eval*/metrics_summary.json"
    plain_train_done="runs/dspcbsd_plus_modified/student_plain/weights/best.pt"
    plain_eval_done="runs/dspcbsd_plus_modified/student_plain_eval*/metrics_summary.json"
    epfa_train_done="runs/dspcbsd_plus_epfa/student_epfa/weights/best.pt"
    epfa_eval_done="runs/dspcbsd_plus_epfa/student_epfa_eval*/metrics_summary.json"
    distill_train_done="runs/dspcbsd_plus_distill_epfa/distill_epfa_fair/weights/best.pt"
    distill_eval_done="runs/dspcbsd_plus_distill_epfa/distill_epfa_fair_eval*/metrics_summary.json"
    ;;
  *)
    echo "unsupported dataset: $dataset" >&2
    exit 1
    ;;
esac

if [[ "$mode" == "full" ]]; then
  run_step_if_missing "$baseline_train_done" "${dataset} baseline train" python experiments/baseline/train.py --config "$baseline_train" || true
  run_step_if_missing "$baseline_eval_done" "${dataset} baseline eval" python experiments/baseline/evaluate.py --config "$baseline_eval" || true
fi
copy_teacher "$teacher_src" "$teacher_dst" || true
run_step_if_missing "$plain_train_done" "${dataset} student_plain train" python experiments/modified_model/train.py --config "$plain_train" || true
run_step_if_missing "$plain_eval_done" "${dataset} student_plain eval" python experiments/modified_model/evaluate.py --config "$plain_eval" || true
run_step_if_missing "$epfa_train_done" "${dataset} student_epfa train" python experiments/modified_model/train.py --config "$epfa_train" || true
run_step_if_missing "$epfa_eval_done" "${dataset} student_epfa eval" python experiments/modified_model/evaluate.py --config "$epfa_eval" || true
run_step_if_missing "$distill_train_done" "${dataset} distill_epfa train" python experiments/distillation/train.py --config "$distill_train" || true
run_step_if_missing "$distill_eval_done" "${dataset} distill_epfa eval" python experiments/distillation/evaluate.py --config "$distill_eval" || true
run_step "${dataset} summarize" python tools/summarize_run.py --pcb-table-root runs --output-dir runs/paper_tables || true

log "PIPELINE COMPLETE ${dataset}"
