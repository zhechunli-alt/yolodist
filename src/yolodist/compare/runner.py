from __future__ import annotations

from pathlib import Path
from time import perf_counter
import csv
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from yolodist.config import load_toml
from yolodist.data.yolo_detection_dataset import YoloDetectionDataset, detection_collate_fn
from yolodist.models.registry import initialize_custom_model_context, register_ultralytics_modules
from yolodist.paths import ROOT
from yolodist.reporting.manifest import write_run_manifest


def run_comparison_train(config_path: Path) -> None:
    config = load_toml(config_path)["comparison"]
    run_dir = ROOT / config["project"] / config["name"]
    write_run_manifest(
        run_dir=run_dir,
        payload={"stage": "comparison_train", "config_path": str(config_path), "comparison": config},
    )
    if config["framework"] == "ultralytics":
        run_ultralytics_train(config)
        return
    run_torchvision_train(config)


def run_comparison_eval(config_path: Path) -> Path:
    config = load_toml(config_path)["comparison"]
    run_dir = ROOT / config["project"] / config["name"]
    write_run_manifest(
        run_dir=run_dir,
        payload={"stage": "comparison_eval", "config_path": str(config_path), "comparison": config},
    )
    if config["framework"] == "ultralytics":
        return run_ultralytics_eval(config)
    return run_torchvision_eval(config)


def run_ultralytics_train(config: dict[str, object]) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for comparison runs.") from exc
    register_ultralytics_modules()
    model = YOLO(str(config["model"]))
    initialize_custom_model_context(model.model)
    model.train(
        data=str(ROOT / str(config["data"])),
        project=str(ROOT / str(config["project"])),
        name=str(config["name"]),
        epochs=int(config["epochs"]),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=str(config["device"]),
        workers=int(config["workers"]),
        seed=int(config["seed"]),
        exist_ok=bool(config["exist_ok"]),
    )


def run_ultralytics_eval(config: dict[str, object]) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for comparison eval.") from exc
    register_ultralytics_modules()
    model = YOLO(str(ROOT / str(config["model"])))
    initialize_custom_model_context(model.model)
    metrics = model.val(
        data=str(ROOT / str(config["data"])),
        split=str(config.get("split", "test")),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=str(config["device"]),
        project=str(ROOT / str(config["project"])),
        name=str(config["name"]),
    )
    save_dir = Path(metrics.save_dir)
    summary_path = save_dir / "metrics_summary.json"
    payload = {}
    if isinstance(getattr(metrics, "results_dict", None), dict):
        payload.update(metrics.results_dict)
    payload["fitness"] = getattr(metrics, "fitness", None)
    payload["speed"] = getattr(metrics, "speed", {})
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def run_torchvision_train(config: dict[str, object]) -> None:
    seed_everything(int(config["seed"]))
    device = torch.device(str(config["device"]))
    data_yaml = ROOT / str(config["data"])
    train_dataset = YoloDetectionDataset(data_yaml, "train", int(config["imgsz"]))
    val_dataset = YoloDetectionDataset(data_yaml, "val", int(config["imgsz"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch"]),
        shuffle=True,
        num_workers=int(config["workers"]),
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch"]),
        shuffle=False,
        num_workers=int(config["workers"]),
        collate_fn=detection_collate_fn,
    )
    model = build_torchvision_model(str(config["model_name"]), train_dataset.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config.get("lr0", 0.005)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(config["epochs"] * 0.7), int(config["epochs"] * 0.9)],
        gamma=0.1,
    )
    run_dir = ROOT / str(config["project"]) / str(config["name"])
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.csv"
    best_map = -1.0
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_map50", "val_map50_95", "val_recall"])
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            running_loss = 0.0
            batch_count = 0
            for images, targets in train_loader:
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu())
                batch_count += 1
            scheduler.step()
            metrics = evaluate_torchvision_model(model, val_loader, device)
            writer.writerow(
                [
                    epoch,
                    f"{running_loss / max(batch_count, 1):.6f}",
                    f"{metrics['metrics/mAP50(B)']:.6f}",
                    f"{metrics['metrics/mAP50-95(B)']:.6f}",
                    f"{metrics['metrics/recall(B)']:.6f}",
                ]
            )
            handle.flush()
            checkpoint = {
                "framework": "torchvision",
                "model_name": str(config["model_name"]),
                "num_classes": train_dataset.num_classes + 1,
                "imgsz": int(config["imgsz"]),
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, weights_dir / "last.pt")
            if metrics["metrics/mAP50-95(B)"] > best_map:
                best_map = metrics["metrics/mAP50-95(B)"]
                torch.save(checkpoint, weights_dir / "best.pt")


def run_torchvision_eval(config: dict[str, object]) -> Path:
    seed_everything(int(config["seed"]))
    device = torch.device(str(config["device"]))
    data_yaml = ROOT / str(config["data"])
    split = str(config.get("split", "test"))
    dataset = YoloDetectionDataset(data_yaml, split, int(config["imgsz"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch"]),
        shuffle=False,
        num_workers=int(config["workers"]),
        collate_fn=detection_collate_fn,
    )
    checkpoint = torch.load(ROOT / str(config["model"]), map_location=device)
    model = build_torchvision_model(str(checkpoint["model_name"]), dataset.num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = evaluate_torchvision_model(model, loader, device, with_speed=True)
    run_dir = ROOT / str(config["project"]) / str(config["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return summary_path


def build_torchvision_model(model_name: str, num_classes_without_background: int):
    from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        fasterrcnn_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
        fcos_resnet50_fpn,
        retinanet_resnet50_fpn_v2,
        ssdlite320_mobilenet_v3_large,
    )

    num_classes = num_classes_without_background + 1
    if model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=num_classes,
        )
    if model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        return fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=num_classes,
        )
    if model_name == "fasterrcnn_resnet50_fpn":
        return fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
        )
    if model_name == "fasterrcnn_resnet50_fpn_v2":
        return fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
        )
    if model_name == "ssdlite320_mobilenet_v3_large":
        return ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=num_classes,
        )
    if model_name == "retinanet_resnet50_fpn_v2":
        return retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
        )
    if model_name == "fcos_resnet50_fpn":
        return fcos_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported torchvision model: {model_name}")


def evaluate_torchvision_model(model, loader, device: torch.device, with_speed: bool = False) -> dict[str, object]:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    inference_time = 0.0
    total_images = 0
    tp = 0
    fp = 0
    fn = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            start = perf_counter()
            outputs = model(images)
            inference_time += perf_counter() - start
            total_images += len(images)
            preds = [
                {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                }
                for output in outputs
            ]
            refs = [
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
                for target in targets
            ]
            metric.update(preds, refs)
            batch_tp, batch_fp, batch_fn = match_predictions_at_iou50(preds, refs)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
    result = metric.compute()
    summary = {
        "metrics/precision(B)": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "metrics/recall(B)": float(result["mar_100"].cpu()),
        "metrics/mAP50(B)": float(result["map_50"].cpu()),
        "metrics/mAP50-95(B)": float(result["map"].cpu()),
        "fitness": float(result["map"].cpu()),
    }
    if with_speed and total_images:
        summary["speed"] = {
            "preprocess": 0.0,
            "inference": 1000.0 * inference_time / total_images,
            "loss": 0.0,
            "postprocess": 0.0,
        }
    return summary


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def match_predictions_at_iou50(preds: list[dict], refs: list[dict], score_threshold: float = 0.25) -> tuple[int, int, int]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for pred, ref in zip(preds, refs):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]
        ref_boxes = ref["boxes"]
        ref_labels = ref["labels"]

        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        matched = torch.zeros(len(ref_boxes), dtype=torch.bool)
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

        image_tp = 0
        image_fp = 0
        for box, label in zip(pred_boxes, pred_labels):
            same_class = (ref_labels == label) & (~matched)
            if same_class.sum() == 0:
                image_fp += 1
                continue
            candidates = ref_boxes[same_class]
            ious = box_iou(box.unsqueeze(0), candidates).squeeze(0)
            best_iou, best_idx = torch.max(ious, dim=0)
            if float(best_iou) >= 0.5:
                ref_indices = torch.nonzero(same_class, as_tuple=False).squeeze(1)
                matched[ref_indices[int(best_idx)]] = True
                image_tp += 1
            else:
                image_fp += 1

        image_fn = int((~matched).sum().item())
        total_tp += image_tp
        total_fp += image_fp
        total_fn += image_fn

    return total_tp, total_fp, total_fn
