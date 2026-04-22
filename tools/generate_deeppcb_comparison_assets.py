#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/workspace/yolodist")
sys.path.insert(0, str(ROOT / "src"))
OUT_DIR = ROOT / "runs/paper_figures/generated"
DATA_YAML = ROOT / "datasets/processed/deeppcb_detection/data.yaml"
IMG_SIZE = 640
SCORE_THRESHOLD = 0.25


def _font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_spec():
    with DATA_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_test_samples():
    spec = _load_spec()
    root = Path(spec.get("path") or DATA_YAML.parent)
    image_dir = root / spec["test"]
    label_dir = root / "labels" / "test"
    samples = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        count = 0
        if label_path.exists():
            count = len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        samples.append((image_path, label_path, count))
    positives = [s for s in samples if s[2] > 0]
    positives.sort(key=lambda x: x[2])
    if len(positives) < 3:
        return positives[:3]
    return [positives[0], positives[len(positives) // 2], positives[-1]]


def _load_gt_boxes(label_path: Path, names: list[str]):
    image = None
    boxes = []
    labels = []
    if not label_path.exists():
        return boxes, labels
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        cls_id, cx, cy, bw, bh = line.split()
        cls = int(cls_id)
        labels.append(names[cls] if cls < len(names) else str(cls))
        boxes.append((float(cx), float(cy), float(bw), float(bh)))
    return boxes, labels


def _draw_gt(image_path: Path, boxes, labels):
    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(image)
    font = _font(16)
    for (cx, cy, bw, bh), label in zip(boxes, labels):
        x1 = (cx - bw / 2.0) * IMG_SIZE
        y1 = (cy - bh / 2.0) * IMG_SIZE
        x2 = (cx + bw / 2.0) * IMG_SIZE
        y2 = (cy + bh / 2.0) * IMG_SIZE
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, y1 + 4), label, fill=(0, 255, 0), font=font)
    return image


def _draw_torchvision_prediction(model_name: str, checkpoint_path: Path, image_path: Path, names: list[str]):
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        ssdlite320_mobilenet_v3_large,
    )

    builders = {
        "ssdlite320_mobilenet_v3_large": lambda: ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=len(names) + 1,
        ),
        "fasterrcnn_mobilenet_v3_large_320_fpn": lambda: fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=len(names) + 1,
        ),
        "fasterrcnn_mobilenet_v3_large_fpn": lambda: fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=len(names) + 1,
        ),
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = builders[model_name]()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
    with torch.inference_mode():
        output = model([tensor])[0]
    draw = ImageDraw.Draw(image)
    font = _font(16)
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        if float(score) < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        cls_idx = int(label.item()) - 1
        cls_name = names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
        draw.text((x1 + 4, y1 + 4), f"{cls_name} {float(score):.2f}", fill=(255, 80, 80), font=font)
    return image


def _draw_student_epfa(image_path: Path, names: list[str]):
    from ultralytics import YOLO
    from yolodist.models.registry import initialize_custom_model_context, register_ultralytics_modules

    register_ultralytics_modules()
    model = YOLO(str(ROOT / "runs/deeppcb_epfa/student_epfa/weights/best.pt"))
    initialize_custom_model_context(model.model)
    results = model.predict(source=str(image_path), imgsz=IMG_SIZE, conf=SCORE_THRESHOLD, verbose=False)
    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(image)
    font = _font(16)
    if results:
        boxes = results[0].boxes
        if boxes is not None:
            for xyxy, conf, cls in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
                x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
                cls_idx = int(cls.item())
                cls_name = names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 180, 0), width=3)
                draw.text((x1 + 4, y1 + 4), f"{cls_name} {float(conf):.2f}", fill=(255, 180, 0), font=font)
    return image


def _panel(samples, names):
    titles = ["GT", "student_epfa", "ssdlite", "frcnn_mnv3_320", "frcnn_mnv3_fpn"]
    models = [
        None,
        ("student_epfa", None),
        ("ssdlite320_mobilenet_v3_large", ROOT / "runs/deeppcb_compare/ssdlite320_mobilenet_v3_large/weights/best.pt"),
        ("fasterrcnn_mobilenet_v3_large_320_fpn", ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn/weights/best.pt"),
        ("fasterrcnn_mobilenet_v3_large_fpn", ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_fpn/weights/best.pt"),
    ]
    cell_w = IMG_SIZE
    cell_h = IMG_SIZE
    pad = 24
    title_h = 42
    row_label_w = 140
    canvas = Image.new(
        "RGB",
        (row_label_w + len(titles) * (cell_w + pad) + pad, len(samples) * (cell_h + title_h + pad) + pad),
        color=(247, 244, 238),
    )
    draw = ImageDraw.Draw(canvas)
    font_title = _font(22)
    font_label = _font(18)

    for col, title in enumerate(titles):
        x = row_label_w + pad + col * (cell_w + pad)
        draw.text((x + 10, 8), title, fill=(20, 20, 20), font=font_title)

    for row, (image_path, label_path, count) in enumerate(samples):
        y = pad + row * (cell_h + title_h + pad) + title_h
        draw.text((16, y + 16), f"{image_path.stem}\nboxes={count}", fill=(30, 30, 30), font=font_label)
        gt_boxes, gt_labels = _load_gt_boxes(label_path, names)
        rendered = [
            _draw_gt(image_path, gt_boxes, gt_labels),
            _draw_student_epfa(image_path, names),
            _draw_torchvision_prediction("ssdlite320_mobilenet_v3_large", models[2][1], image_path, names),
            _draw_torchvision_prediction("fasterrcnn_mobilenet_v3_large_320_fpn", models[3][1], image_path, names),
            _draw_torchvision_prediction("fasterrcnn_mobilenet_v3_large_fpn", models[4][1], image_path, names),
        ]
        for col, img in enumerate(rendered):
            x = row_label_w + pad + col * (cell_w + pad)
            canvas.paste(img, (x, y))

    out = OUT_DIR / "deeppcb_additional_qualitative_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def _plot_curves():
    configs = {
        "ssdlite": ROOT / "runs/deeppcb_compare/ssdlite320_mobilenet_v3_large/results.csv",
        "fasterrcnn_mnv3_320": ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn/results.csv",
        "fasterrcnn_mnv3_fpn": ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_fpn/results.csv",
        "fasterrcnn_r50_fpn": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn/results.csv",
        "fasterrcnn_r50_fpn_v2": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn_v2/results.csv",
    }
    metrics = {}
    for name, path in configs.items():
        epochs, map95, recall = [], [], []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                map95.append(float(row["val_map50_95"]))
                recall.append(float(row["val_recall"]))
        metrics[name] = (epochs, map95, recall)

    plt.figure(figsize=(10, 5))
    for name, (epochs, map95, _) in metrics.items():
        plt.plot(epochs, map95, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("val mAP50-95")
    plt.title("DeepPCB Additional Comparison Training Curves")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "deeppcb_additional_train_curve_map5095.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    for name, (epochs, _, recall) in metrics.items():
        plt.plot(epochs, recall, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("val Recall")
    plt.title("DeepPCB Additional Comparison Recall Curves")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "deeppcb_additional_train_curve_recall.png", dpi=220)
    plt.close()


def main():
    spec = _load_spec()
    names = spec["names"]
    samples = _load_test_samples()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _panel(samples, names)
    _plot_curves()
    print("Generated DeepPCB comparison qualitative and curve assets.")


if __name__ == "__main__":
    main()
