#!/usr/bin/env bash
set -u

ROOT="/root/workspace/yolodist"
LOG_DIR="$ROOT/runs/overnight_logs"
LOG_FILE="$LOG_DIR/missing_distill_plain.log"
mkdir -p "$LOG_DIR"

source /root/workspace/.venv/bin/activate
cd "$ROOT" || exit 1

log() {
  printf '[%s] %s\n' "$(date -u +%F'T'%T'Z')" "$*" | tee -a "$LOG_FILE"
}

run_with_retry() {
  local label="$1"
  shift
  local attempt=1
  while [ "$attempt" -le 2 ]; do
    log "START ${label} (attempt ${attempt})"
    if "$@" >>"$LOG_FILE" 2>&1; then
      log "DONE ${label} (attempt ${attempt})"
      return 0
    fi
    log "FAIL ${label} (attempt ${attempt})"
    attempt=$((attempt + 1))
  done
  log "GIVEUP ${label}"
  return 1
}

is_dataset_complete() {
  local best_dst="$1"
  local eval_project="$2"
  local eval_name_prefix="$3"
  if [ ! -f "$best_dst" ]; then
    return 1
  fi
  if find "$eval_project" -maxdepth 2 -type f -path "*/${eval_name_prefix}*/metrics_summary.json" | grep -q .; then
    return 0
  fi
  return 1
}

copy_weight_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    cp -f "$src" "$dst"
    log "COPIED $(basename "$src") -> $dst"
  else
    log "MISSING best weight: $src"
  fi
}

run_dataset() {
  local name="$1"
  local train_cfg="$2"
  local eval_cfg="$3"
  local best_src="$4"
  local best_dst="$5"
  local eval_project="$6"
  local eval_name_prefix="$7"

  if is_dataset_complete "$best_dst" "$eval_project" "$eval_name_prefix"; then
    log "SKIP ${name} because weight and eval metrics already exist"
    return 0
  fi

  run_with_retry "${name} train" python experiments/distillation/train.py --config "$train_cfg"
  copy_weight_if_exists "$best_src" "$best_dst"
  if [ -f "$best_src" ]; then
    run_with_retry "${name} eval" python experiments/distillation/evaluate.py --config "$eval_cfg"
  else
    log "SKIP ${name} eval because best weight is missing"
  fi
}

log "==== missing distill_plain补权重开始 ===="
run_dataset \
  "DeepPCB distill_plain" \
  "configs/train/distillation_deeppcb_plain.toml" \
  "configs/eval/distillation_deeppcb_plain.toml" \
  "runs/deeppcb_distill/distill_plain/weights/best.pt" \
  "weights/best_deeppcb_distill_plain.pt" \
  "runs/deeppcb_distill" \
  "distill_plain_eval"

run_dataset \
  "PKU distill_plain" \
  "configs/train/distillation_pku_market_pcb_plain.toml" \
  "configs/eval/distillation_pku_market_pcb_plain.toml" \
  "runs/pku_market_pcb_distill/distill_plain/weights/best.pt" \
  "weights/best_pku_distill_plain.pt" \
  "runs/pku_market_pcb_distill" \
  "distill_plain_eval"

run_dataset \
  "DsPCBSD+ distill_plain" \
  "configs/train/distillation_dspcbsd_plus_plain.toml" \
  "configs/eval/distillation_dspcbsd_plus_plain.toml" \
  "runs/dspcbsd_plus_distill/distill_plain/weights/best.pt" \
  "weights/best_dspcbsd_distill_plain.pt" \
  "runs/dspcbsd_plus_distill" \
  "distill_plain_eval"

log "==== missing distill_plain补权重结束 ===="
