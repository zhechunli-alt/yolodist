from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any

from yolodist.config import load_toml
from yolodist.models.registry import register_ultralytics_modules
from yolodist.paths import ROOT
from yolodist.reporting.manifest import write_run_manifest


def run_distillation(config_path: Path) -> None:
    config = load_toml(config_path)["distill"]
    alpha = float(config.get("distill_alpha", 0.0))
    temperature = float(config.get("distill_temperature", 1.0))
    enable_kd_loss = bool(config.get("enable_kd_loss", True))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    register_ultralytics_modules()

    teacher_weights = ROOT / config["teacher_weights"]
    student_model = ROOT / config["student_model"]
    prepared_data = ROOT / config["prepared_data"]
    pseudo_dataset_dir = ROOT / config["pseudo_dataset_dir"]
    run_dir = ROOT / config["project"] / config["name"]

    if not teacher_weights.exists():
        raise FileNotFoundError(f"Teacher weights not found: {teacher_weights}")
    if not student_model.exists():
        raise FileNotFoundError(f"Student model not found: {student_model}")
    if not prepared_data.exists():
        raise FileNotFoundError(f"Prepared dataset yaml not found: {prepared_data}")
    if "student_pretrained" in config:
        student_pretrained = ROOT / config["student_pretrained"]
        if not student_pretrained.exists():
            raise FileNotFoundError(f"Student pretrained checkpoint not found: {student_pretrained}")

    write_run_manifest(
        run_dir=run_dir,
        payload={
            "stage": "distill",
            "config_path": str(config_path),
            "distill": config,
            "kd_active": enable_kd_loss and alpha > 0.0,
            "kd_type": "response_mse",
        },
    )
    build_pseudo_dataset(prepared_data, pseudo_dataset_dir)
    stats = pseudo_label_train_split(
        teacher_weights=teacher_weights,
        pseudo_dataset_dir=pseudo_dataset_dir,
        conf=float(config["teacher_conf"]),
        device=str(config["device"]),
        strategy=str(config.get("label_strategy", "fill_empty")),
        iou_thresh=float(config.get("teacher_iou_threshold", 0.5)),
        max_det=int(config.get("teacher_max_det", 300)),
    )
    (run_dir / "pseudo_label_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    student = YOLO(str(student_model))
    if enable_kd_loss and alpha > 0.0:
        attach_response_kd_loss(
            student_yolo=student,
            teacher_weights=teacher_weights,
            alpha=alpha,
            temperature=temperature,
        )
    train_kwargs = {
        "data": str(pseudo_dataset_dir / "data.yaml"),
        "project": str(ROOT / config["project"]),
        "name": config["name"],
        "epochs": int(config["epochs"]),
        "imgsz": int(config["imgsz"]),
        "batch": int(config["batch"]),
        "device": str(config["device"]),
        "workers": int(config["workers"]),
        "seed": int(config["seed"]),
        "exist_ok": bool(config["exist_ok"]),
    }
    if "student_pretrained" in config:
        train_kwargs["pretrained"] = str(ROOT / config["student_pretrained"])
    student.train(
        **train_kwargs
    )


def build_pseudo_dataset(original_yaml: Path, pseudo_dataset_dir: Path) -> None:
    original_dataset_dir = original_yaml.parent
    if pseudo_dataset_dir.exists():
        shutil.rmtree(pseudo_dataset_dir)

    (pseudo_dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (pseudo_dataset_dir / "labels").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src_image_dir = original_dataset_dir / "images" / split
        src_label_dir = original_dataset_dir / "labels" / split
        dst_image_dir = pseudo_dataset_dir / "images" / split
        dst_label_dir = pseudo_dataset_dir / "labels" / split
        link_or_copy_tree(src_image_dir, dst_image_dir)
        shutil.copytree(src_label_dir, dst_label_dir)

    shutil.copy2(original_yaml, pseudo_dataset_dir / "data.yaml")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def pseudo_label_train_split(
    teacher_weights: Path,
    pseudo_dataset_dir: Path,
    conf: float,
    device: str,
    strategy: str,
    iou_thresh: float,
    max_det: int,
) -> dict[str, int]:
    from ultralytics import YOLO
    register_ultralytics_modules()

    model = YOLO(str(teacher_weights))
    train_image_dir = pseudo_dataset_dir / "images" / "train"
    train_label_dir = pseudo_dataset_dir / "labels" / "train"
    stats = {
        "total_images": 0,
        "images_with_original_labels": 0,
        "images_with_teacher_labels": 0,
        "teacher_boxes_added": 0,
        "images_skipped_by_strategy": 0,
    }
    if strategy not in {"fill_empty", "merge_teacher"}:
        raise ValueError(f"Unsupported label_strategy: {strategy}")

    for image_path in sorted(train_image_dir.iterdir()):
        if not image_path.is_file():
            continue
        stats["total_images"] += 1
        label_path = train_label_dir / f"{image_path.stem}.txt"
        original_lines = read_label_lines(label_path)
        has_original = bool(original_lines)
        if has_original:
            stats["images_with_original_labels"] += 1
        if strategy == "fill_empty" and has_original:
            stats["images_skipped_by_strategy"] += 1
            continue

        results = model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
            device=device,
            max_det=max_det,
        )
        teacher_boxes = predictions_to_boxes(results)
        merged_boxes = merge_boxes(
            original_boxes=parse_yolo_lines(original_lines),
            teacher_boxes=teacher_boxes,
            iou_thresh=iou_thresh,
            keep_original=True,
        )
        if merged_boxes:
            out_lines = [box_to_yolo_line(box) for box in merged_boxes]
            label_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            if teacher_boxes:
                stats["images_with_teacher_labels"] += 1
            teacher_added = max(0, len(merged_boxes) - len(parse_yolo_lines(original_lines)))
            stats["teacher_boxes_added"] += teacher_added
    return stats


def predictions_to_boxes(results) -> list[tuple[int, float, float, float, float]]:
    boxes_out: list[tuple[int, float, float, float, float]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xywhn = boxes.xywhn.tolist()
        classes = boxes.cls.tolist()
        for cls_id, box in zip(classes, xywhn, strict=False):
            boxes_out.append((int(cls_id), box[0], box[1], box[2], box[3]))
    return boxes_out


def read_label_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_yolo_lines(lines: list[str]) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def box_to_yolo_line(box: tuple[int, float, float, float, float]) -> str:
    cls_id, x, y, w, h = box
    return f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def merge_boxes(
    original_boxes: list[tuple[int, float, float, float, float]],
    teacher_boxes: list[tuple[int, float, float, float, float]],
    iou_thresh: float,
    keep_original: bool,
) -> list[tuple[int, float, float, float, float]]:
    merged = list(original_boxes) if keep_original else []
    for teacher_box in teacher_boxes:
        if any(iou_xywh(teacher_box, base_box) >= iou_thresh for base_box in merged):
            continue
        merged.append(teacher_box)
    return merged


def iou_xywh(box_a: tuple[int, float, float, float, float], box_b: tuple[int, float, float, float, float]) -> float:
    _, ax, ay, aw, ah = box_a
    _, bx, by, bw, bh = box_b
    ax1, ay1, ax2, ay2 = to_xyxy(ax, ay, aw, ah)
    bx1, by1, bx2, by2 = to_xyxy(bx, by, bw, bh)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(aw, 0.0) * max(ah, 0.0)
    area_b = max(bw, 0.0) * max(bh, 0.0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    half_w = w / 2.0
    half_h = h / 2.0
    return x - half_w, y - half_h, x + half_w, y + half_h


def attach_response_kd_loss(student_yolo, teacher_weights: Path, alpha: float, temperature: float) -> None:
    """Attach response-level KD loss into student model.loss.

    The merged objective becomes:
    L_total = (1 - alpha) * L_det + alpha * L_kd
    """
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"distill_alpha must be in [0, 1], got {alpha}")
    if temperature <= 0.0:
        raise ValueError(f"distill_temperature must be > 0, got {temperature}")

    teacher_yolo = YOLO(str(teacher_weights))
    teacher_model = teacher_yolo.model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    student_model = student_yolo.model
    base_loss_fn = student_model.loss
    state: dict[str, Any] = {"teacher_device": None}

    def kd_wrapped_loss(batch, preds=None):
        det_loss, det_items = base_loss_fn(batch, preds)
        student_preds = preds if preds is not None else student_model(batch["img"])
        kd_loss = response_kd_loss(
            student_preds=student_preds,
            teacher_model=teacher_model,
            images=batch["img"],
            temperature=temperature,
            state=state,
            mse_fn=F.mse_loss,
        )
        total = (1.0 - alpha) * det_loss + alpha * kd_loss
        return total, det_items

    student_model.loss = kd_wrapped_loss


def response_kd_loss(student_preds, teacher_model, images, temperature: float, state: dict[str, Any], mse_fn) -> "torch.Tensor":
    import torch

    image_device = images.device
    if state.get("teacher_device") != image_device:
        teacher_model.to(image_device)
        state["teacher_device"] = image_device

    with torch.no_grad():
        teacher_preds = teacher_model(images)

    student_tensors = extract_pred_tensors(student_preds)
    teacher_tensors = extract_pred_tensors(teacher_preds)
    if not student_tensors or not teacher_tensors:
        return torch.zeros((), device=image_device)

    kd = torch.zeros((), device=image_device)
    pair_count = 0
    for s, t in zip(student_tensors, teacher_tensors, strict=False):
        if s.shape != t.shape:
            continue
        kd = kd + mse_fn(s / temperature, t / temperature)
        pair_count += 1
    if pair_count == 0:
        return torch.zeros((), device=image_device)
    return kd * (temperature**2) / float(pair_count)


def extract_pred_tensors(preds) -> list:
    tensors: list = []
    if isinstance(preds, (list, tuple)):
        for item in preds:
            tensors.extend(extract_pred_tensors(item))
        return tensors
    if hasattr(preds, "shape") and getattr(preds, "ndim", 0) == 4:
        tensors.append(preds)
    return tensors
