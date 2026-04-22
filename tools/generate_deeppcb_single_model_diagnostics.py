#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import yaml

ROOT = Path("/root/workspace/yolodist")
sys.path.insert(0, str(ROOT / "src"))

from yolodist.compare.runner import build_torchvision_model
from yolodist.data.yolo_detection_dataset import YoloDetectionDataset, detection_collate_fn

DATA_YAML = ROOT / "datasets/processed/deeppcb_detection/data.yaml"
OUT_DIR = ROOT / "runs/paper_figures/generated"
IMG_SIZE = 640
SCORE_THRESHOLD = 0.25

TORCHVISION_MODELS = [
    ("ssdlite320_mobilenet_v3_large", ROOT / "runs/deeppcb_compare/ssdlite320_mobilenet_v3_large/weights/best.pt"),
    ("retinanet_r50_fpn", ROOT / "runs/deeppcb_compare/retinanet_r50_fpn/weights/best.pt"),
    ("fcos_r50_fpn", ROOT / "runs/deeppcb_compare/fcos_r50_fpn/weights/best.pt"),
    ("fasterrcnn_mnv3_320_fpn", ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn/weights/best.pt"),
    ("fasterrcnn_mnv3_fpn", ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_fpn/weights/best.pt"),
    ("fasterrcnn_r50_fpn", ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn/weights/best.pt"),
    ("fasterrcnn_r50_fpn_v2", ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn_v2/weights/best.pt"),
]

STUDENT_EPFA_ARTIFACTS = {
    "pred": ROOT / "runs/deeppcb_epfa/student_epfa_eval-2/val_batch0_pred.jpg",
    "pr": ROOT / "runs/deeppcb_epfa/student_epfa_eval-2/BoxPR_curve.png",
    "cm": ROOT / "runs/deeppcb_epfa/student_epfa_eval-2/confusion_matrix_normalized.png",
}

MODEL_NAME_MAP = {
    "ssdlite320_mobilenet_v3_large": "ssdlite320_mobilenet_v3_large",
    "retinanet_r50_fpn": "retinanet_resnet50_fpn_v2",
    "fcos_r50_fpn": "fcos_resnet50_fpn",
    "fasterrcnn_mnv3_320_fpn": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mnv3_fpn": "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_r50_fpn": "fasterrcnn_resnet50_fpn",
    "fasterrcnn_r50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
}


def _font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def load_spec():
    with DATA_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(raw_names) -> list[str]:
    if isinstance(raw_names, dict):
        return [raw_names[key] for key in sorted(raw_names)]
    return list(raw_names)


def choose_sample():
    spec = load_spec()
    root = Path(spec.get("path") or DATA_YAML.parent)
    image_dir = root / spec["test"]
    label_dir = root / "labels" / "test"
    candidates = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            candidates.append((image_path, label_path, len(lines)))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0]


def draw_gt(image_path: Path, label_path: Path, names: list[str]):
    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(image)
    font = _font(16)
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        cls_id, cx, cy, bw, bh = line.split()
        cls_id = int(cls_id)
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1 = (cx - bw / 2.0) * IMG_SIZE
        y1 = (cy - bh / 2.0) * IMG_SIZE
        x2 = (cx + bw / 2.0) * IMG_SIZE
        y2 = (cy + bh / 2.0) * IMG_SIZE
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, y1 + 4), names[cls_id], fill=(0, 255, 0), font=font)
    image.save(OUT_DIR / "deeppcb_ground_truth_reference.png")


def load_torchvision_checkpoint(model_key: str):
    ckpt_path = dict(TORCHVISION_MODELS)[model_key]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_name = checkpoint.get("model_name", MODEL_NAME_MAP[model_key])
    spec = load_spec()
    num_classes = int(spec["nc"])
    model = build_torchvision_model(model_name, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def save_prediction_image(model_key: str, model, image_path: Path, names: list[str], device: torch.device):
    image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.array(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    with torch.inference_mode():
        output = model([tensor.to(device)])[0]
    draw = ImageDraw.Draw(image)
    font = _font(16)
    for box, score, label in zip(output["boxes"].detach().cpu(), output["scores"].detach().cpu(), output["labels"].detach().cpu()):
        if float(score) < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        cls_idx = int(label.item()) - 1
        cls_name = names[cls_idx] if 0 <= cls_idx < len(names) else str(cls_idx)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
        draw.text((x1 + 4, y1 + 4), f"{cls_name} {float(score):.2f}", fill=(255, 80, 80), font=font)
    image.save(OUT_DIR / f"deeppcb_{model_key}_single_pred.png")


def evaluate_model(model_key: str, model, device: torch.device, names: list[str]):
    dataset = YoloDetectionDataset(DATA_YAML, "test", IMG_SIZE)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=detection_collate_fn)

    num_classes = len(names)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    pr_records = []
    gt_count = 0

    with torch.inference_mode():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()
                gt_boxes = target["boxes"].detach().cpu()
                gt_labels = target["labels"].detach().cpu()
                gt_count += len(gt_boxes)

                # confusion matrix with background row/col
                unmatched_gt = torch.ones(len(gt_boxes), dtype=torch.bool)
                order = torch.argsort(pred_scores, descending=True)
                for idx in order.tolist():
                    score = float(pred_scores[idx])
                    if score < SCORE_THRESHOLD:
                        continue
                    box = pred_boxes[idx]
                    pred_label = int(pred_labels[idx].item()) - 1
                    if len(gt_boxes) == 0:
                        cm[num_classes, pred_label] += 1
                        pr_records.append((score, 0))
                        continue
                    ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
                    best_iou, best_idx = torch.max(ious, dim=0)
                    best_idx = int(best_idx)
                    gt_label = int(gt_labels[best_idx].item()) - 1
                    if float(best_iou) >= 0.5 and unmatched_gt[best_idx]:
                        unmatched_gt[best_idx] = False
                        cm[gt_label, pred_label] += 1
                        pr_records.append((score, 1 if gt_label == pred_label else 0))
                    else:
                        cm[num_classes, pred_label] += 1
                        pr_records.append((score, 0))
                for gt_idx, still_unmatched in enumerate(unmatched_gt.tolist()):
                    if still_unmatched:
                        gt_label = int(gt_labels[gt_idx].item()) - 1
                        cm[gt_label, num_classes] += 1

    save_confusion_matrix(model_key, cm, names)
    save_pr_curve(model_key, pr_records, gt_count)


def save_confusion_matrix(model_key: str, cm: np.ndarray, names: list[str]):
    labels = names + ["background"]
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    plt.figure(figsize=(8, 7))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(f"DeepPCB {model_key} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"deeppcb_{model_key}_confusion_matrix.png", dpi=220)
    plt.close()


def save_pr_curve(model_key: str, pr_records: list[tuple[float, int]], gt_count: int):
    if not pr_records or gt_count == 0:
        return
    pr_records.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for _, is_tp in pr_records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / gt_count)
    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, color="#1f4e79", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"DeepPCB {model_key} PR Curve")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"deeppcb_{model_key}_pr_curve.png", dpi=220)
    plt.close()


def copy_student_epfa_assets():
    mapping = {
        STUDENT_EPFA_ARTIFACTS["pred"]: OUT_DIR / "deeppcb_student_epfa_single_pred.png",
        STUDENT_EPFA_ARTIFACTS["pr"]: OUT_DIR / "deeppcb_student_epfa_pr_curve.png",
        STUDENT_EPFA_ARTIFACTS["cm"]: OUT_DIR / "deeppcb_student_epfa_confusion_matrix.png",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copy2(src, dst)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    names = normalize_names(spec["names"])
    image_path, label_path, _ = choose_sample()
    draw_gt(image_path, label_path, names)
    copy_student_epfa_assets()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_key, _ in TORCHVISION_MODELS:
        try:
            model = load_torchvision_checkpoint(model_key).to(device)
        except FileNotFoundError as exc:
            print(f"SKIP {model_key}: {exc}")
            continue
        save_prediction_image(model_key, model, image_path, names, device)
        evaluate_model(model_key, model, device, names)
    print("Generated DeepPCB single-model diagnostics for torchvision comparison models.")


if __name__ == "__main__":
    main()
