#!/usr/bin/env bash
set -u

cd /root/workspace/yolodist || exit 1

mkdir -p runs/overnight_logs
log_file="runs/overnight_logs/pcb_overnight_mainlines.log"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$log_file"
}

is_running() {
  local pattern="$1"
  pgrep -af "$pattern" >/dev/null 2>&1
}

has_match() {
  local pattern="$1"
  compgen -G "$pattern" >/dev/null
}

maybe_start() {
  local dataset="$1"
  local mode="$2"
  local done_marker="$3"
  local active_pattern="$4"

  if has_match "$done_marker"; then
    log "SKIP ${dataset}: pipeline complete (${done_marker})"
    return 0
  fi

  if is_running "$active_pattern"; then
    log "HOLD ${dataset}: active process matched"
    return 0
  fi

  log "LAUNCH ${dataset} (${mode})"
  nohup bash -lc "cd /root/workspace/yolodist && ./tools/run_dataset_mainline.sh ${dataset} ${mode}" \
    >> "runs/overnight_logs/${dataset}_overnight.log" 2>&1 &
}

while true; do
  if [[ -f runs/pku_market_pcb_baseline/baseline_plain/weights/best.pt ]]; then
    pku_mode="after-baseline"
  else
    pku_mode="full"
  fi

  maybe_start \
    "pku_market_pcb" \
    "$pku_mode" \
    "runs/pku_market_pcb_distill_epfa/distill_epfa_fair_eval*/metrics_summary.json" \
    "tools/run_dataset_mainline.sh pku_market_pcb|baseline_pku_market_pcb.toml|modified_model_pku_market_pcb_plain.toml|modified_model_pku_market_pcb_epfa.toml|distillation_pku_market_pcb_epfa.toml"

  maybe_start \
    "dspcbsd_plus" \
    "full" \
    "runs/dspcbsd_plus_distill_epfa/distill_epfa_fair_eval*/metrics_summary.json" \
    "tools/run_dataset_mainline.sh dspcbsd_plus|baseline_dspcbsd_plus.toml|modified_model_dspcbsd_plus_plain.toml|modified_model_dspcbsd_plus_epfa.toml|distillation_dspcbsd_plus_epfa.toml"

  sleep 60
done
