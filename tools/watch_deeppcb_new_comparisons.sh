#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/yolodist
source /root/workspace/.venv/bin/activate

log_dir="runs/overnight_logs"
mkdir -p "$log_dir"
log_file="$log_dir/deeppcb_new_comparisons.log"

check_eval() {
  local dir="$1"
  compgen -G "$dir" > /dev/null
}

run_eval_once() {
  local config="$1"
  local eval_glob="$2"
  if compgen -G "$eval_glob" > /dev/null; then
    return 0
  fi
  python experiments/comparison/evaluate.py --config "$config" >> "$log_file" 2>&1
}

while true; do
  ts="$(date -u +%FT%TZ)"
  echo "[$ts] watcher tick" >> "$log_file"

  if [[ -f runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn/weights/best.pt ]] && ! pgrep -af 'comparison/train.py --config configs/train/comparison_deeppcb_fasterrcnn_mnv3_320.toml' >/dev/null; then
    run_eval_once \
      "configs/eval/comparison_deeppcb_fasterrcnn_mnv3_320.toml" \
      "runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn_eval*/metrics_summary.json"
  fi

  if [[ -f runs/deeppcb_compare/fasterrcnn_mnv3_fpn/weights/best.pt ]] && ! pgrep -af 'comparison/train.py --config configs/train/comparison_deeppcb_fasterrcnn_mnv3_fpn.toml' >/dev/null; then
    run_eval_once \
      "configs/eval/comparison_deeppcb_fasterrcnn_mnv3_fpn.toml" \
      "runs/deeppcb_compare/fasterrcnn_mnv3_fpn_eval*/metrics_summary.json"
  fi

  if [[ -f runs/deeppcb_compare/ssdlite320_mobilenet_v3_large/weights/best.pt ]] && ! pgrep -af 'comparison/train.py --config configs/train/comparison_deeppcb_ssdlite320.toml' >/dev/null; then
    run_eval_once \
      "configs/eval/comparison_deeppcb_ssdlite320.toml" \
      "runs/deeppcb_compare/ssdlite320_mobilenet_v3_large_eval*/metrics_summary.json"
  fi

  if compgen -G "runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn_eval*/metrics_summary.json" > /dev/null && \
     compgen -G "runs/deeppcb_compare/fasterrcnn_mnv3_fpn_eval*/metrics_summary.json" > /dev/null && \
     compgen -G "runs/deeppcb_compare/ssdlite320_mobilenet_v3_large_eval*/metrics_summary.json" > /dev/null; then
    echo "[$ts] all new comparison evals complete" >> "$log_file"
    exit 0
  fi

  sleep 30
done
