# Experiment Flow

This file explains how the current codebase is intended to be used once you move to a server.

## 1. Prepare data

Run:

```bash
python3 tools/prepare_mvtec_detection.py --config configs/data/mvtec_detection.toml
```

What it does:

- scans `datasets/mvtec/`
- reads anomaly masks
- converts masks into bounding boxes
- writes a YOLO-format dataset under `datasets/processed/mvtec_detection/`

Why it exists:

- MVTec is not directly trainable by YOLO detection code

## 2. Train a baseline

Run:

```bash
python3 experiments/baseline/train.py
```

What it does:

- loads `configs/train/baseline.toml`
- reads model weights from `weights/`
- starts a standard Ultralytics training run

Why it exists:

- you need a standard detector as the comparison baseline

## 3. Train a student or modified model

Run:

```bash
python3 experiments/modified_model/train.py
```

What it does now:

- loads a dedicated lightweight student model yaml:
  - `configs/models/yolo11_student.yaml`
- can optionally initialize from pretrained weights via config
- default config is PPLA-enhanced student
- plain ablation config is also available:
  - `configs/train/modified_model_plain.toml`

What should happen later:

- extend this student with thesis-oriented modules such as Ghost/BiFPN variants

## 4. Run distillation

Run:

```bash
python3 experiments/distillation/train.py
```

Plain ablation:

```bash
python3 experiments/distillation/train.py --config configs/train/distillation_plain.toml
```

What it does:

- loads teacher weights
- supports two label strategies:
  - `fill_empty`: only fill labels for empty samples
  - `merge_teacher`: merge teacher predictions into training labels with IoU filtering
- trains the student with that dataset

Why it exists:

- this is the first feasible distillation path without deeply modifying Ultralytics internals
- each run also writes `run_manifest.json` and pseudo-label stats for experiment reproducibility

## 5. Evaluate

Run:

```bash
python3 experiments/baseline/evaluate.py
python3 experiments/modified_model/evaluate.py
python3 experiments/distillation/evaluate.py
```

What it does:

- runs Ultralytics validation
- writes `metrics_summary.json` into the evaluation run directory

## 6. Summarize results

Run:

```bash
python3 tools/summarize_run.py runs/baseline/your_run_name
```

What it does:

- reads `results.csv` or `metrics_summary.json`
- extracts the most useful metrics
- writes a compact markdown summary for later thesis writing
