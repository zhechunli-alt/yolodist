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

Optional NEU-DET path:

```bash
python3 tools/prepare_neudet_detection.py --config configs/data/neudet_detection.toml
```

PCB datasets:

```bash
python3 tools/download_deeppcb_from_github.py --out datasets/deeppcb/PCBData
python3 tools/prepare_pcb_detection.py --dataset deeppcb --src datasets/deeppcb --out datasets/processed/deeppcb_detection
python3 tools/prepare_pcb_detection.py --dataset pku_market_pcb --src datasets/pku_market_pcb --out datasets/processed/pku_market_pcb_detection
python3 tools/prepare_pcb_detection.py --dataset dspcbsd_plus --src datasets/dspcbsd_plus --out datasets/processed/dspcbsd_plus_detection
```

2. Run preflight checks before training:

```bash
python3 tools/preflight_check.py
```

PCB detailed checks:

```bash
python3 tools/preflight_check.py --train-config configs/train/baseline_deeppcb.toml --data-config datasets/processed/deeppcb_detection/data.yaml
```

3. Put model weights in `weights/`:
   - `weights/yolo11n.pt`
   - `weights/teacher.pt`
   - `weights/student.pt` if you already have one

4. Run baseline training:

```bash
python3 experiments/baseline/train.py
```

NEU-DET baseline:

```bash
python3 experiments/baseline/train.py --config configs/train/baseline_neudet.toml
```

5. Run modified-model training (PPLA student by default):

```bash
python3 experiments/modified_model/train.py
```

NEU-DET modified-model:

```bash
python3 experiments/modified_model/train.py --config configs/train/modified_model_neudet.toml
```

Optional plain student ablation:

```bash
python3 experiments/modified_model/train.py --config configs/train/modified_model_plain.toml
```

6. Run distillation:

```bash
python3 experiments/distillation/train.py
```

NEU-DET distillation:

```bash
python3 experiments/distillation/train.py --config configs/train/distillation_neudet.toml
```

PCB quick pipeline:

```bash
bash tools/run_pcb_pipeline.sh --dataset deeppcb --mode all
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

NEU-DET evaluation:

```bash
python3 experiments/baseline/evaluate.py --config configs/eval/baseline_neudet.toml
python3 experiments/modified_model/evaluate.py --config configs/eval/modified_model_neudet.toml
python3 experiments/distillation/evaluate.py --config configs/eval/distillation_neudet.toml
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

Generate PCB paper tables:

```bash
python3 tools/summarize_run.py --pcb-table-root runs --output-dir runs/paper_tables
```

## Dependencies

The current environment does not have training dependencies installed. To actually train, install:

```bash
pip install -r requirements.txt
```

If you continue the local YOLOX comparison pipeline, prefer:

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e external/YOLOX
```

Detailed restart and environment notes are recorded in:

- [docs/服务器恢复与环境说明.md](/root/workspace/yolodist/docs/服务器恢复与环境说明.md)

## Backend service (thread C)

1. Sync latest run weights to backend standard names (optional):

```bash
python3 tools/sync_standard_weights.py
```

2. If you only need API/frontend smoke tests now, bootstrap standard names from one `.pt` (optional):

```bash
python3 tools/bootstrap_backend_weights.py --source weights/yolo11n.pt
```

3. Run backend preflight checks:

```bash
python3 tools/backend_preflight.py
```

4. Start Flask API:

```bash
python3 -m service.backend.app
```

Available endpoints:

1. `GET /api/v1/health`
2. `POST /api/v1/infer`
3. `GET /api/v1/models`
4. `POST /api/v1/models/switch`
5. `GET /api/v1/stats/summary`

Model registry follows standard weights names in `weights/`:

1. `best_teacher.pt`
2. `best_student_plain.pt`
3. `best_student_ppla.pt`
4. `best_student_distill.pt`

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
- PCB migration adds an EPFA-Lite student YAML (`configs/models/yolo11_student_epfa.yaml`) intended for P3/P4/P5 edge-aware ablation.

## Project records

- `docs/WORKLOG.md`: engineering changes and decisions already made
- `docs/NEXT_STEPS.md`: recommended implementation order from this point
- `docs/PCB_CLASS_MAPPING.md`: unified PCB class naming and alias rules
- `docs/PCB迁移与EPFA说明.md`: PCB migration rationale, EPFA design, and experiment notes
