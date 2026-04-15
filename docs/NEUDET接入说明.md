# NEU-DET 接入说明

## 接入改动清单

- 新增数据准备实现：`src/yolodist/data/neudet_detection.py`
- 新增准备脚本：`tools/prepare_neudet_detection.py`
- 新增数据配置：`configs/data/neudet_detection.toml`
- 新增训练配置：
  - `configs/train/baseline_neudet.toml`
  - `configs/train/modified_model_neudet.toml`
  - `configs/train/distillation_neudet.toml`
- 新增评估配置：
  - `configs/eval/baseline_neudet.toml`
  - `configs/eval/modified_model_neudet.toml`
  - `configs/eval/distillation_neudet.toml`

## 期望的原始数据目录

默认按以下结构读取：

```text
datasets/neudet/
├── IMAGES/
│   ├── *.jpg
│   └── ...
└── ANNOTATIONS/
    ├── *.xml
    └── ...
```

如果目录名不同，可以在 `configs/data/neudet_detection.toml` 中修改 `image_dir_name` 和 `annotation_dir_name`。

## 支持的标注格式

- `voc_xml`：Pascal VOC 风格 XML
- `yolo_txt`：YOLO txt
- `auto`：自动检测，优先 XML，再尝试 TXT

## 跑通命令链

```bash
python3 tools/prepare_neudet_detection.py --config configs/data/neudet_detection.toml
python3 experiments/baseline/train.py --config configs/train/baseline_neudet.toml
cp -f runs/neudet_baseline/yolo11n_baseline/weights/best.pt weights/teacher_neudet.pt
python3 experiments/modified_model/train.py --config configs/train/modified_model_neudet.toml
python3 experiments/distillation/train.py --config configs/train/distillation_neudet.toml
python3 experiments/baseline/evaluate.py --config configs/eval/baseline_neudet.toml
python3 experiments/modified_model/evaluate.py --config configs/eval/modified_model_neudet.toml
python3 experiments/distillation/evaluate.py --config configs/eval/distillation_neudet.toml
```

## 说明

- NEU-DET 本身是工业表面缺陷检测数据集，更适合直接使用 `mAP` 指标。
- 现有 baseline / modified / distill 入口无需改动，只需要切换数据和输出配置。
