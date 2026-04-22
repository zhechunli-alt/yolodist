#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/yolodist

out_dir="runs/paper_figures"
mkdir -p "$out_dir"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
  fi
}

# DeepPCB
copy_if_exists "runs/deeppcb_baseline/baseline_plain_eval-2/val_batch0_pred.jpg" "$out_dir/deeppcb_baseline_pred.jpg"
copy_if_exists "runs/deeppcb_modified/student_plain_eval-2/val_batch0_pred.jpg" "$out_dir/deeppcb_student_plain_pred.jpg"
copy_if_exists "runs/deeppcb_epfa/student_epfa_eval-2/val_batch0_pred.jpg" "$out_dir/deeppcb_student_epfa_pred.jpg"
copy_if_exists "runs/deeppcb_distill_epfa/distill_epfa_fair_eval-2/val_batch0_pred.jpg" "$out_dir/deeppcb_distill_epfa_fair_pred.jpg"
copy_if_exists "runs/deeppcb_distill_epfa_tuned/distill_epfa_tuned_eval-2/val_batch0_pred.jpg" "$out_dir/deeppcb_distill_epfa_tuned_pred.jpg"
copy_if_exists "runs/deeppcb_epfa/student_epfa_eval-2/confusion_matrix_normalized.png" "$out_dir/deeppcb_student_epfa_confusion.png"
copy_if_exists "runs/deeppcb_epfa/student_epfa_eval-2/BoxPR_curve.png" "$out_dir/deeppcb_student_epfa_pr_curve.png"
copy_if_exists "runs/deeppcb_epfa/student_epfa/results.png" "$out_dir/deeppcb_student_epfa_train_curve.png"

# PKU
copy_if_exists "runs/pku_market_pcb_baseline/baseline_plain/val_batch0_pred.jpg" "$out_dir/pku_baseline_pred.jpg"
copy_if_exists "runs/pku_market_pcb_modified/student_plain_eval-3/val_batch0_pred.jpg" "$out_dir/pku_student_plain_pred.jpg"
copy_if_exists "runs/pku_market_pcb_epfa/student_epfa_eval-2/val_batch0_pred.jpg" "$out_dir/pku_student_epfa_pred.jpg"
copy_if_exists "runs/pku_market_pcb_distill_epfa/distill_epfa_fair_eval-2/val_batch0_pred.jpg" "$out_dir/pku_distill_epfa_fair_pred.jpg"
copy_if_exists "runs/pku_market_pcb_distill_epfa_tuned/distill_epfa_tuned_eval-3/val_batch0_pred.jpg" "$out_dir/pku_distill_epfa_tuned_pred.jpg"
copy_if_exists "runs/pku_market_pcb_epfa/student_epfa_eval-2/confusion_matrix_normalized.png" "$out_dir/pku_student_epfa_confusion.png"
copy_if_exists "runs/pku_market_pcb_epfa/student_epfa_eval-2/BoxPR_curve.png" "$out_dir/pku_student_epfa_pr_curve.png"
copy_if_exists "runs/pku_market_pcb_epfa/student_epfa/results.png" "$out_dir/pku_student_epfa_train_curve.png"

# DsPCBSD+
copy_if_exists "runs/dspcbsd_plus_baseline/baseline_plain_eval-2/val_batch0_pred.jpg" "$out_dir/dspcbsd_baseline_pred.jpg"
copy_if_exists "runs/dspcbsd_plus_modified/student_plain_eval-2/val_batch0_pred.jpg" "$out_dir/dspcbsd_student_plain_pred.jpg"
copy_if_exists "runs/dspcbsd_plus_epfa/student_epfa_eval-2/val_batch0_pred.jpg" "$out_dir/dspcbsd_student_epfa_pred.jpg"
copy_if_exists "runs/dspcbsd_plus_distill_epfa/distill_epfa_fair_eval-2/val_batch0_pred.jpg" "$out_dir/dspcbsd_distill_epfa_fair_pred.jpg"
copy_if_exists "runs/dspcbsd_plus_distill_epfa_tuned/distill_epfa_tuned_eval-2/val_batch0_pred.jpg" "$out_dir/dspcbsd_distill_epfa_tuned_pred.jpg"
copy_if_exists "runs/dspcbsd_plus_epfa/student_epfa_eval-2/confusion_matrix_normalized.png" "$out_dir/dspcbsd_student_epfa_confusion.png"
copy_if_exists "runs/dspcbsd_plus_epfa/student_epfa_eval-2/BoxPR_curve.png" "$out_dir/dspcbsd_student_epfa_pr_curve.png"
copy_if_exists "runs/dspcbsd_plus_epfa/student_epfa/results.png" "$out_dir/dspcbsd_student_epfa_train_curve.png"

cat > "$out_dir/README.md" <<'MD'
# Paper Figures

## Suggested qualitative comparison figures

- `deeppcb_baseline_pred.jpg`
- `deeppcb_student_plain_pred.jpg`
- `deeppcb_student_epfa_pred.jpg`
- `deeppcb_distill_epfa_fair_pred.jpg`
- `deeppcb_distill_epfa_tuned_pred.jpg`

- `pku_baseline_pred.jpg`
- `pku_student_plain_pred.jpg`
- `pku_student_epfa_pred.jpg`
- `pku_distill_epfa_fair_pred.jpg`
- `pku_distill_epfa_tuned_pred.jpg`

- `dspcbsd_baseline_pred.jpg`
- `dspcbsd_student_plain_pred.jpg`
- `dspcbsd_student_epfa_pred.jpg`
- `dspcbsd_distill_epfa_fair_pred.jpg`
- `dspcbsd_distill_epfa_tuned_pred.jpg`

## Suggested diagnostic figures

- `*_student_epfa_confusion.png`
- `*_student_epfa_pr_curve.png`
- `*_student_epfa_train_curve.png`

## Suggested paper usage

1. One qualitative comparison panel per dataset:
   baseline vs student_plain vs student_epfa vs distill_epfa_fair
2. One confusion matrix for the strongest deployable student line.
3. One PR curve for the strongest deployable student line.
4. One training curve (`results.png`) to show convergence stability.
MD

printf 'Collected figures into %s\n' "$out_dir"
