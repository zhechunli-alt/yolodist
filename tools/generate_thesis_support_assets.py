#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/workspace/yolodist")
sys.path.insert(0, str(ROOT / "src"))

from yolodist.compare.runner import build_torchvision_model  # noqa: E402
from yolodist.models.registry import initialize_custom_model_context, register_ultralytics_modules  # noqa: E402


OUT_TABLE = ROOT / "runs/paper_tables"
OUT_FIG = ROOT / "runs/paper_figures/generated"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATASETS = {
    "DeepPCB": ROOT / "datasets/processed/deeppcb_detection/data.yaml",
    "PKU-Market-PCB": ROOT / "datasets/processed/pku_market_pcb_detection/data.yaml",
    "DsPCBSD+": ROOT / "datasets/processed/dspcbsd_plus_detection/data.yaml",
}

BENCHMARK_MODELS = [
    {
        "name": "baseline",
        "kind": "ultralytics",
        "weight": ROOT / "runs/deeppcb_baseline/baseline_plain/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_baseline/baseline_plain_eval-2/metrics_summary.json",
    },
    {
        "name": "student_plain",
        "kind": "ultralytics",
        "weight": ROOT / "runs/deeppcb_modified/student_plain/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_modified/student_plain_eval-2/metrics_summary.json",
    },
    {
        "name": "student_epfa",
        "kind": "ultralytics",
        "weight": ROOT / "runs/deeppcb_epfa/student_epfa/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_epfa/student_epfa_eval-2/metrics_summary.json",
    },
    {
        "name": "distill_epfa_fair",
        "kind": "ultralytics",
        "weight": ROOT / "runs/deeppcb_distill_epfa/distill_epfa_fair/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_distill_epfa/distill_epfa_fair_eval-2/metrics_summary.json",
    },
    {
        "name": "distill_epfa_tuned",
        "kind": "ultralytics",
        "weight": ROOT / "runs/deeppcb_distill_epfa_tuned/distill_epfa_tuned/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_distill_epfa_tuned/distill_epfa_tuned_eval-2/metrics_summary.json",
    },
    {
        "name": "fasterrcnn_mnv3_fpn",
        "kind": "torchvision",
        "weight": ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_fpn/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_fpn_eval/metrics_summary.json",
    },
    {
        "name": "fasterrcnn_r50_fpn",
        "kind": "torchvision",
        "weight": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn_eval/metrics_summary.json",
    },
    {
        "name": "fasterrcnn_r50_fpn_v2",
        "kind": "torchvision",
        "weight": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn_v2/weights/best.pt",
        "metrics": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn_v2_eval/metrics_summary.json",
    },
]

TORCHVISION_MODEL_NAMES = {
    "fasterrcnn_mnv3_fpn": "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_r50_fpn": "fasterrcnn_resnet50_fpn",
    "fasterrcnn_r50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
}


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_stats():
    rows = []
    class_plots = {}
    for dataset, yaml_path in DATASETS.items():
        spec = load_yaml(yaml_path)
        root = Path(spec.get("path") or yaml_path.parent)
        names = normalize_names(spec["names"])
        split_dirs = {split: root / spec[split] for split in ["train", "val", "test"]}
        label_dirs = {split: root / "labels" / split for split in ["train", "val", "test"]}
        class_counter = Counter()
        totals = {}
        for split in ["train", "val", "test"]:
            image_count = len([p for p in split_dirs[split].glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
            box_count = 0
            label_count = 0
            for label_path in label_dirs[split].glob("*.txt"):
                lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if lines:
                    label_count += 1
                for line in lines:
                    cls_id = int(line.split()[0])
                    class_counter[(split, cls_id)] += 1
                    box_count += 1
            totals[split] = {
                "images": image_count,
                "images_with_labels": label_count,
                "boxes": box_count,
            }
        for split in ["train", "val", "test"]:
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "images": totals[split]["images"],
                    "images_with_labels": totals[split]["images_with_labels"],
                    "boxes": totals[split]["boxes"],
                    "num_classes": len(names),
                }
            )
        class_plots[dataset] = {
            "names": names,
            "counts": [class_counter[("train", i)] for i in range(len(names))],
        }

    csv_path = OUT_TABLE / "pcb_dataset_stats.csv"
    md_path = OUT_TABLE / "pcb_dataset_stats.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "images", "images_with_labels", "boxes", "num_classes"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# PCB 数据集统计",
        "",
        "| 数据集 | 划分 | 图像数 | 有标签图像数 | 框数 | 类别数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['images']} | {row['images_with_labels']} | {row['boxes']} | {row['num_classes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (dataset, plot) in zip(axes, class_plots.items()):
        ax.bar(plot["names"], plot["counts"], color="#3b7a57")
        ax.set_title(dataset)
        ax.set_ylabel("Train boxes")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "pcb_dataset_class_distribution.png", dpi=220)
    plt.close(fig)


def load_ultralytics_model(weight: Path):
    from ultralytics import YOLO

    register_ultralytics_modules()
    yolo = YOLO(str(weight))
    initialize_custom_model_context(yolo.model)
    return yolo.model


def load_torchvision_model(weight: Path):
    ckpt = torch.load(weight, map_location="cpu")
    model_name = ckpt.get("model_name")
    model = build_torchvision_model(model_name, ckpt["num_classes"] - 1)
    model.load_state_dict(ckpt["state_dict"])
    return model


def benchmark_speed(model, kind: str, imgsz: int = 640, warmup: int = 10, runs: int = 50):
    model = model.to(DEVICE).eval()
    x = torch.rand(1, 3, imgsz, imgsz, device=DEVICE)
    with torch.inference_mode():
        for _ in range(warmup):
            if kind == "ultralytics":
                _ = model(x)
            else:
                _ = model([x[0]])
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            if kind == "ultralytics":
                _ = model(x)
            else:
                _ = model([x[0]])
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    ms = 1000.0 * elapsed / runs
    fps = 1000.0 / ms if ms > 0 else 0.0
    return ms, fps


def benchmark_models():
    rows = []
    for cfg in BENCHMARK_MODELS:
        if not cfg["weight"].exists() or not cfg["metrics"].exists():
            continue
        metrics = load_json(cfg["metrics"])
        if cfg["kind"] == "ultralytics":
            model = load_ultralytics_model(cfg["weight"])
        else:
            model = load_torchvision_model(cfg["weight"])
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        bench_ms, bench_fps = benchmark_speed(model, cfg["kind"])
        rows.append(
            {
                "model": cfg["name"],
                "params_m": params_m,
                "precision": float(metrics.get("metrics/precision(B)", 0.0)),
                "recall": float(metrics.get("metrics/recall(B)", 0.0)),
                "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
                "map5095": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
                "eval_inference_ms": float(metrics.get("speed", {}).get("inference", 0.0)),
                "bench_inference_ms": bench_ms,
                "bench_fps": bench_fps,
            }
        )

    csv_path = OUT_TABLE / "deeppcb_benchmark_models.csv"
    md_path = OUT_TABLE / "deeppcb_benchmark_models.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "params_m",
                "precision",
                "recall",
                "map50",
                "map5095",
                "eval_inference_ms",
                "bench_inference_ms",
                "bench_fps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# DeepPCB 模型效率与性能表",
        "",
        "| 模型 | Params(M) | Precision | Recall | mAP50 | mAP50-95 | 统一测速 ms | 统一测速 FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['params_m']:.3f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['map50']:.4f} | {row['map5095']:.4f} | {row['bench_inference_ms']:.3f} | {row['bench_fps']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    keep = [r for r in rows if r["model"] in {"baseline", "student_epfa", "fasterrcnn_mnv3_fpn", "fasterrcnn_r50_fpn", "fasterrcnn_r50_fpn_v2"}]
    plt.figure(figsize=(9, 5))
    for row in keep:
        plt.scatter(row["params_m"], row["map5095"], s=130)
        plt.annotate(row["model"], (row["params_m"], row["map5095"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Params (M)")
    plt.ylabel("mAP50-95")
    plt.title("DeepPCB Parameter-Accuracy Trade-off")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "deeppcb_param_accuracy_tradeoff_strong.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 5))
    for row in keep:
        plt.scatter(row["bench_fps"], row["map5095"], s=130)
        plt.annotate(row["model"], (row["bench_fps"], row["map5095"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Benchmark FPS")
    plt.ylabel("mAP50-95")
    plt.title("DeepPCB Speed-Accuracy Trade-off")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "deeppcb_speed_accuracy_tradeoff_strong.png", dpi=220)
    plt.close()


def _font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def montage(title: str, image_paths: list[Path], labels: list[str], out_path: Path, cols: int = 3):
    images = [Image.open(p).convert("RGB") for p in image_paths if p.exists()]
    labels = [l for p, l in zip(image_paths, labels) if p.exists()]
    if not images:
        return
    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    pad = 24
    header_h = 60
    label_h = 34
    canvas = Image.new(
        "RGB",
        (cols * (max_w + pad) + pad, header_h + rows * (max_h + label_h + pad) + pad),
        color=(247, 244, 238),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), title, fill=(20, 20, 20), font=_font(26))
    for i, (im, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (max_w + pad)
        y = header_h + pad + r * (max_h + label_h + pad)
        canvas.paste(im.resize((max_w, max_h)), (x, y))
        draw.rectangle([x, y + max_h, x + max_w, y + max_h + label_h], fill=(233, 229, 220))
        draw.text((x + 8, y + max_h + 6), label, fill=(30, 30, 30), font=_font(18))
    canvas.save(out_path)


def make_montages():
    montage(
        "PCB Student EPFA Confusion Matrices",
        [
            ROOT / "runs/paper_figures/deeppcb_student_epfa_confusion.png",
            ROOT / "runs/paper_figures/pku_student_epfa_confusion.png",
            ROOT / "runs/paper_figures/dspcbsd_student_epfa_confusion.png",
        ],
        ["DeepPCB", "PKU-Market-PCB", "DsPCBSD+"],
        OUT_FIG / "pcb_confusion_montage.png",
        cols=3,
    )
    montage(
        "PCB Student EPFA PR Curves",
        [
            ROOT / "runs/paper_figures/deeppcb_student_epfa_pr_curve.png",
            ROOT / "runs/paper_figures/pku_student_epfa_pr_curve.png",
            ROOT / "runs/paper_figures/dspcbsd_student_epfa_pr_curve.png",
        ],
        ["DeepPCB", "PKU-Market-PCB", "DsPCBSD+"],
        OUT_FIG / "pcb_prcurve_montage.png",
        cols=3,
    )
    montage(
        "PCB Student EPFA Training Curves",
        [
            ROOT / "runs/paper_figures/deeppcb_student_epfa_train_curve.png",
            ROOT / "runs/paper_figures/pku_student_epfa_train_curve.png",
            ROOT / "runs/paper_figures/dspcbsd_student_epfa_train_curve.png",
        ],
        ["DeepPCB", "PKU-Market-PCB", "DsPCBSD+"],
        OUT_FIG / "pcb_traincurve_montage.png",
        cols=3,
    )


def write_asset_notes():
    path = OUT_TABLE / "thesis_support_assets.md"
    lines = [
        "# 论文支撑素材补充清单",
        "",
        "本次新增的是不依赖长时间重训练的论文支撑素材，重点覆盖：数据集统计、统一 benchmark、模型效率图、诊断图拼图。",
        "",
        "## 新增表格",
        "",
        "- `pcb_dataset_stats.csv` / `pcb_dataset_stats.md`：三个 PCB 数据集的图像数、标注图像数、框数、类别数统计",
        "- `deeppcb_benchmark_models.csv` / `deeppcb_benchmark_models.md`：DeepPCB 关键模型统一 benchmark 表",
        "",
        "## 新增图像",
        "",
        "- `pcb_dataset_class_distribution.png`：三个 PCB 数据集训练集类别分布",
        "- `deeppcb_param_accuracy_tradeoff_strong.png`：强基线与主方法的参数-精度权衡图",
        "- `deeppcb_speed_accuracy_tradeoff_strong.png`：强基线与主方法的速度-精度权衡图",
        "- `pcb_confusion_montage.png`：三数据集混淆矩阵拼图",
        "- `pcb_prcurve_montage.png`：三数据集 PR 曲线拼图",
        "- `pcb_traincurve_montage.png`：三数据集训练曲线拼图",
        "",
        "## 适合论文使用的位置",
        "",
        "1. 数据集章节：`pcb_dataset_stats.*` 与 `pcb_dataset_class_distribution.png`",
        "2. 主结果章节：`deeppcb_benchmark_models.*` 与 `deeppcb_param_accuracy_tradeoff_strong.png`",
        "3. 部署/效率讨论：`deeppcb_speed_accuracy_tradeoff_strong.png`",
        "4. 诊断分析章节：三种 montage 图",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    dataset_stats()
    benchmark_models()
    make_montages()
    write_asset_notes()
    print("Generated thesis support assets.")


if __name__ == "__main__":
    main()
