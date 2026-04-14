# YOLODist

YOLO11 industrial defect detection and distillation project scaffold.

## Project focus

This repository implements the model side of the thesis direction in a feasible way:

1. Convert MVTec AD from anomaly masks to a YOLO detection dataset.
2. Train a baseline YOLO11 detector.
3. Train a lighter student model.
4. Distill the student with teacher-guided pseudo labels.

The original proposal also includes prompt-based data generation, TensorRT deployment, and a full web system. Those are valid extensions, but they are not the first milestone here. The first goal is a reproducible model pipeline that can train, evaluate, and iterate.

## Current technical choices

- Dataset task: anomaly detection is converted to object detection through mask-to-box conversion.
- Default label mode: `binary` (`defect` vs background), which is the safest starting point for MVTec.
- Distillation strategy: teacher pseudo-label distillation.
  - This is more feasible than feature-level distillation at the current stage.
  - It avoids early coupling to modified Ultralytics internals.
- Modified model experiment: now uses a dedicated lightweight student YAML.

## Repository layout

```text
yolodist/
├── configs/
│   ├── data/
│   └── train/
├── datasets/
│   └── mvtec/
├── experiments/
│   ├── baseline/
│   ├── distillation/
│   └── modified_model/
├── runs/
├── src/
│   └── yolodist/
├── tools/
├── ultralytics-src/
└── weights/
```

## Workflow

1. Prepare a detection dataset from MVTec:

```bash
python3 tools/prepare_mvtec_detection.py --config configs/data/mvtec_detection.toml
```

2. Run preflight checks before training:

```bash
python3 tools/preflight_check.py
```

3. Put model weights in `weights/`:
   - `weights/yolo11n.pt`
   - `weights/teacher.pt`
   - `weights/student.pt` if you already have one

4. Run baseline training:

```bash
python3 experiments/baseline/train.py
```

5. Run modified-model training (PPLA student by default):

```bash
python3 experiments/modified_model/train.py
```

Optional plain student ablation:

```bash
python3 experiments/modified_model/train.py --config configs/train/modified_model_plain.toml
```

6. Run distillation:

```bash
python3 experiments/distillation/train.py
```

Optional plain student distillation ablation:

```bash
python3 experiments/distillation/train.py --config configs/train/distillation_plain.toml
```

7. Evaluate a run:

```bash
python3 experiments/baseline/evaluate.py
python3 experiments/modified_model/evaluate.py
python3 experiments/distillation/evaluate.py
```

Plain ablation evaluation:

```bash
python3 experiments/modified_model/evaluate.py --config configs/eval/modified_model_plain.toml
python3 experiments/distillation/evaluate.py --config configs/eval/distillation_plain.toml
```

8. Summarize the resulting run directory:

```bash
python3 tools/summarize_run.py runs/baseline/your_run_name
```

## Dependencies

The current environment does not have training dependencies installed. To actually train, install:

```bash
pip install -r requirements.txt
```

## Notes on feasibility

- MVTec AD is not a native detection dataset. The conversion script creates bounding boxes from masks and repartitions anomaly samples for a detection workflow.
- This means the generated train/val/test split is for this project pipeline, not the official MVTec benchmark protocol.
- The current distillation pipeline supports:
  - `label_strategy = "fill_empty"`: teacher only labels unlabeled samples
  - `label_strategy = "merge_teacher"`: teacher boxes are merged with original labels by IoU filtering
- `distill_temperature` and `distill_alpha` now directly participate in response-level KD loss:
  - `L_total = (1 - alpha) * L_det + alpha * L_kd(T)`
  - current `L_kd` uses MSE on teacher/student multi-scale detection outputs with temperature scaling
- The default modified model is a PPLA-enhanced student YAML (`configs/models/yolo11_student.yaml`) and a plain ablation YAML is also provided (`configs/models/yolo11_student_plain.yaml`).

## Project records

- `docs/WORKLOG.md`: engineering changes and decisions already made
- `docs/NEXT_STEPS.md`: recommended implementation order from this point
