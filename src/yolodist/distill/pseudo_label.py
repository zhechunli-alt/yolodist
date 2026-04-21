from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any

import yaml

from yolodist.config import load_toml
from yolodist.models.registry import initialize_custom_model_context, register_ultralytics_modules
from yolodist.paths import ROOT
from yolodist.reporting.manifest import write_run_manifest


def run_distillation(config_path: Path) -> None:
    config = load_toml(config_path)["distill"]
    alpha = float(config.get("distill_alpha", 0.0))
    temperature = float(config.get("distill_temperature", 1.0))
    enable_kd_loss = bool(config.get("enable_kd_loss", True))
    enable_feature_kd_loss = bool(config.get("enable_feature_kd_loss", False))
    feature_kd_alpha = float(config.get("feature_kd_alpha", 0.0))
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
            "kd_type": str(config.get("kd_strategy", "response_mse")),
            "pseudo_dataset_dir": str(pseudo_dataset_dir),
        },
    )
    build_pseudo_dataset(prepared_data, pseudo_dataset_dir)
    validate_pseudo_dataset_layout(prepared_data, pseudo_dataset_dir)
    stats = pseudo_label_train_split(
        teacher_weights=teacher_weights,
        pseudo_dataset_dir=pseudo_dataset_dir,
        conf=float(config["teacher_conf"]),
        device=str(config["device"]),
        strategy=str(config.get("label_strategy", "fill_empty")),
        iou_thresh=float(config.get("teacher_iou_threshold", 0.5)),
        max_det=int(config.get("teacher_max_det", 300)),
    )
    clear_label_caches(pseudo_dataset_dir)
    (run_dir / "pseudo_label_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    validate_pseudo_label_effect(
        original_yaml=prepared_data,
        pseudo_dataset_dir=pseudo_dataset_dir,
        stats=stats,
    )
    integrity = validate_pseudo_dataset(prepared_data, pseudo_dataset_dir, stats)
    (run_dir / "pseudo_dataset_integrity.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not integrity["ok"]:
        raise RuntimeError(f"Pseudo dataset integrity check failed: {integrity}")

    student = YOLO(str(student_model))
    initialize_custom_model_context(student.model)
    if enable_kd_loss and alpha > 0.0:
        attach_response_kd_loss(
            student_yolo=student,
            teacher_weights=teacher_weights,
            alpha=alpha,
            temperature=temperature,
            kd_strategy=str(config.get("kd_strategy", "response_mse")),
            enable_feature_kd_loss=enable_feature_kd_loss,
            feature_kd_alpha=feature_kd_alpha,
            feature_kd_layers=list(config.get("feature_kd_layers", [15, 18, 21])),
            teacher_feature_kd_layers=list(config.get("teacher_feature_kd_layers", config.get("feature_kd_layers", [15, 18, 21]))),
            cls_kd_alpha=float(config.get("cls_kd_alpha", 0.25)),
            loc_kd_alpha=float(config.get("loc_kd_alpha", 1.0)),
            bg_weight=float(config.get("bg_weight", 0.05)),
            mask_expand_ratio=float(config.get("mask_expand_ratio", 0.0)),
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
    if "optimizer" in config:
        train_kwargs["optimizer"] = str(config["optimizer"])
    if "lr0" in config:
        train_kwargs["lr0"] = float(config["lr0"])
    if "momentum" in config:
        train_kwargs["momentum"] = float(config["momentum"])
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
        shutil.copytree(src_image_dir, dst_image_dir)
        shutil.copytree(src_label_dir, dst_label_dir)
    clear_label_caches(pseudo_dataset_dir)

    dataset_yaml = yaml.safe_load(original_yaml.read_text(encoding="utf-8"))
    dataset_yaml["path"] = str(pseudo_dataset_dir.resolve())
    (pseudo_dataset_dir / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def validate_pseudo_dataset_layout(original_yaml: Path, pseudo_dataset_dir: Path) -> None:
    original_dataset_dir = original_yaml.parent.resolve()
    pseudo_dataset_dir = pseudo_dataset_dir.resolve()
    if pseudo_dataset_dir == original_dataset_dir:
        raise RuntimeError("Pseudo dataset directory must differ from original dataset directory.")

    pseudo_yaml = pseudo_dataset_dir / "data.yaml"
    dataset_yaml = yaml.safe_load(pseudo_yaml.read_text(encoding="utf-8"))
    yaml_path = Path(str(dataset_yaml.get("path", ""))).resolve()
    if yaml_path != pseudo_dataset_dir:
        raise RuntimeError(
            f"Pseudo data.yaml path mismatch: expected {pseudo_dataset_dir}, got {yaml_path}"
        )

    for split in ("train", "val", "test"):
        src_image_dir = original_dataset_dir / "images" / split
        dst_image_dir = pseudo_dataset_dir / "images" / split
        src_label_dir = original_dataset_dir / "labels" / split
        dst_label_dir = pseudo_dataset_dir / "labels" / split

        if not dst_image_dir.exists() or not dst_label_dir.exists():
            raise RuntimeError(f"Pseudo dataset missing split directories for {split}")
        if dst_image_dir.is_symlink():
            raise RuntimeError(f"Pseudo image dir for {split} must be copied, not symlinked")

        src_images = sorted(p.name for p in src_image_dir.iterdir() if p.is_file())
        dst_images = sorted(p.name for p in dst_image_dir.iterdir() if p.is_file())
        src_labels = sorted(p.name for p in src_label_dir.iterdir() if p.is_file())
        dst_labels = sorted(p.name for p in dst_label_dir.iterdir() if p.is_file())
        if src_images != dst_images:
            raise RuntimeError(f"Pseudo image file list mismatch for split {split}")
        if src_labels != dst_labels:
            raise RuntimeError(f"Pseudo label file list mismatch for split {split}")

        sample_images = dst_images[:5]
        for image_name in sample_images:
            sample_path = dst_image_dir / image_name
            if sample_path.is_symlink():
                raise RuntimeError(f"Pseudo image file {sample_path} must be copied, not symlinked")


def validate_pseudo_label_effect(original_yaml: Path, pseudo_dataset_dir: Path, stats: dict[str, int]) -> None:
    if stats["total_images"] <= 0:
        raise RuntimeError("Pseudo labeling found zero training images.")

    if stats["images_with_teacher_labels"] <= 0:
        raise RuntimeError("Teacher produced zero pseudo-labeled images; distillation setup is likely ineffective.")

    if stats["teacher_boxes_added"] <= 0:
        raise RuntimeError("Teacher added zero boxes; pseudo labels did not change the training labels.")

    original_train_dir = original_yaml.parent / "labels" / "train"
    pseudo_train_dir = pseudo_dataset_dir / "labels" / "train"
    changed_files = 0
    for pseudo_label_path in sorted(pseudo_train_dir.glob("*.txt")):
        original_label_path = original_train_dir / pseudo_label_path.name
        original_text = original_label_path.read_text(encoding="utf-8") if original_label_path.exists() else ""
        pseudo_text = pseudo_label_path.read_text(encoding="utf-8")
        if original_text != pseudo_text:
            changed_files += 1
    if changed_files <= 0:
        raise RuntimeError("Pseudo label files are identical to the original labels; distillation would collapse to standard training.")


def clear_label_caches(pseudo_dataset_dir: Path) -> None:
    for cache_path in (pseudo_dataset_dir / "labels").rglob("*.cache"):
        cache_path.unlink(missing_ok=True)


def validate_pseudo_dataset(
    original_yaml: Path,
    pseudo_dataset_dir: Path,
    stats: dict[str, int],
) -> dict[str, Any]:
    dataset_yaml = yaml.safe_load((pseudo_dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    expected_path = str(pseudo_dataset_dir.resolve())
    yaml_path = str(dataset_yaml.get("path", ""))
    image_train_dir = pseudo_dataset_dir / "images" / "train"
    label_train_dir = pseudo_dataset_dir / "labels" / "train"
    sample_image = next((p for p in sorted(image_train_dir.iterdir()) if p.is_file()), None)
    sample_label = next((p for p in sorted(label_train_dir.iterdir()) if p.is_file() and p.suffix == ".txt"), None)
    cache_files = sorted(str(p.relative_to(pseudo_dataset_dir)) for p in (pseudo_dataset_dir / "labels").rglob("*.cache"))
    teacher_box_ratio = 0.0
    if stats.get("total_images", 0) > 0:
        teacher_box_ratio = float(stats.get("teacher_boxes_added", 0)) / float(stats["total_images"])
    integrity = {
        "ok": True,
        "original_yaml": str(original_yaml),
        "pseudo_dataset_dir": str(pseudo_dataset_dir),
        "yaml_path": yaml_path,
        "expected_yaml_path": expected_path,
        "yaml_path_matches": yaml_path == expected_path,
        "train_image_dir_exists": image_train_dir.exists(),
        "train_label_dir_exists": label_train_dir.exists(),
        "sample_image_is_symlink": sample_image.is_symlink() if sample_image else None,
        "sample_label_is_symlink": sample_label.is_symlink() if sample_label else None,
        "remaining_cache_files": cache_files,
        "teacher_boxes_added": int(stats.get("teacher_boxes_added", 0)),
        "teacher_box_ratio": teacher_box_ratio,
    }
    integrity["ok"] = bool(
        integrity["yaml_path_matches"]
        and integrity["train_image_dir_exists"]
        and integrity["train_label_dir_exists"]
        and integrity["sample_image_is_symlink"] is False
        and integrity["sample_label_is_symlink"] is False
        and not integrity["remaining_cache_files"]
    )
    return integrity


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


def attach_response_kd_loss(
    student_yolo,
    teacher_weights: Path,
    alpha: float,
    temperature: float,
    kd_strategy: str = "response_mse",
    enable_feature_kd_loss: bool = False,
    feature_kd_alpha: float = 0.0,
    feature_kd_layers: list[int] | None = None,
    teacher_feature_kd_layers: list[int] | None = None,
    cls_kd_alpha: float = 0.25,
    loc_kd_alpha: float = 1.0,
    bg_weight: float = 0.05,
    mask_expand_ratio: float = 0.0,
) -> None:
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
    initialize_custom_model_context(teacher_model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    student_model = student_yolo.model
    initialize_custom_model_context(student_model)
    base_loss_fn = student_model.loss
    state: dict[str, Any] = {"teacher_device": None}
    feature_state = build_feature_kd_state(
        student_model=student_model,
        teacher_model=teacher_model,
        student_layers=feature_kd_layers or [15, 18, 21],
        teacher_layers=teacher_feature_kd_layers or feature_kd_layers or [15, 18, 21],
    )

    def kd_wrapped_loss(batch, preds=None):
        det_loss, det_items = base_loss_fn(batch, preds)
        student_preds = student_model(batch["img"])
        kd_loss = response_kd_loss(
            student_preds=student_preds,
            teacher_model=teacher_model,
            images=batch["img"],
            temperature=temperature,
            state=state,
            mse_fn=F.mse_loss,
            batch=batch,
            kd_strategy=kd_strategy,
            cls_kd_alpha=cls_kd_alpha,
            loc_kd_alpha=loc_kd_alpha,
            bg_weight=bg_weight,
            mask_expand_ratio=mask_expand_ratio,
        )
        total = (1.0 - alpha) * det_loss + alpha * kd_loss
        if enable_feature_kd_loss and feature_kd_alpha > 0.0:
            total = total + feature_kd_alpha * feature_kd_loss(
                feature_state,
                mse_fn=F.mse_loss,
                device=batch["img"].device,
                batch=batch,
                bg_weight=bg_weight,
                mask_expand_ratio=mask_expand_ratio,
            )
        return total, det_items

    student_model.loss = kd_wrapped_loss


def response_kd_loss(
    student_preds,
    teacher_model,
    images,
    temperature: float,
    state: dict[str, Any],
    mse_fn,
    batch: dict[str, Any] | None = None,
    kd_strategy: str = "response_mse",
    cls_kd_alpha: float = 0.25,
    loc_kd_alpha: float = 1.0,
    bg_weight: float = 0.05,
    mask_expand_ratio: float = 0.0,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

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

    if kd_strategy == "loc_fg_distill":
        detect_module = getattr(teacher_model, "model", [teacher_model])[-1]
        reg_max = int(getattr(detect_module, "reg_max", 16))
        nc = int(getattr(detect_module, "nc", 1))
        reg_channels = reg_max * 4
        kd = torch.zeros((), device=image_device)
        pair_count = 0
        for s, t in zip(student_tensors, teacher_tensors, strict=False):
            if s.shape != t.shape:
                continue
            if s.shape[1] < reg_channels + nc:
                continue
            height, width = int(s.shape[2]), int(s.shape[3])
            fg_mask = build_fg_mask(
                batch=batch,
                batch_size=int(s.shape[0]),
                height=height,
                width=width,
                device=image_device,
                expand_ratio=mask_expand_ratio,
            )
            cls_mask = mask_with_background_weight(fg_mask, bg_weight=bg_weight)
            s_box = s[:, :reg_channels].reshape(s.shape[0], 4, reg_max, height, width)
            t_box = t[:, :reg_channels].reshape(t.shape[0], 4, reg_max, height, width)
            s_box_log = F.log_softmax(s_box / temperature, dim=2)
            t_box_prob = F.softmax(t_box / temperature, dim=2)
            loc_map = F.kl_div(s_box_log, t_box_prob, reduction="none").sum(dim=2).mean(dim=1, keepdim=True)
            loc_loss = weighted_spatial_mean(loc_map * (temperature**2), fg_mask, bg_weight=0.0)

            s_cls = torch.sigmoid(s[:, reg_channels : reg_channels + nc] / temperature)
            t_cls = torch.sigmoid(t[:, reg_channels : reg_channels + nc] / temperature)
            cls_map = (s_cls - t_cls).pow(2).mean(dim=1, keepdim=True)
            cls_loss = weighted_spatial_mean(cls_map * (temperature**2), cls_mask, bg_weight=0.0)

            kd = kd + (loc_kd_alpha * loc_loss) + (cls_kd_alpha * cls_loss)
            pair_count += 1
        if pair_count == 0:
            return torch.zeros((), device=image_device)
        return kd / float(pair_count)

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


def build_feature_kd_state(student_model, teacher_model, student_layers: list[int], teacher_layers: list[int]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "student": {},
        "teacher": {},
        "hooks": [],
        "student_layers": list(student_layers),
        "teacher_layers": list(teacher_layers),
    }

    for store_key, model, layer_ids in (
        ("student", student_model, student_layers),
        ("teacher", teacher_model, teacher_layers),
    ):
        module_list = getattr(model, "model", None)
        if module_list is None:
            continue
        for layer_id in layer_ids:
            if not (0 <= layer_id < len(module_list)):
                continue
            module = module_list[layer_id]

            def save_output(_, __, output, *, _store_key=store_key, _layer_id=layer_id):
                state[_store_key][_layer_id] = output

            state["hooks"].append(module.register_forward_hook(save_output))
    return state


def feature_kd_loss(
    feature_state: dict[str, Any],
    mse_fn,
    device,
    batch: dict[str, Any] | None = None,
    bg_weight: float = 0.05,
    mask_expand_ratio: float = 0.0,
) -> "torch.Tensor":
    import torch

    student_layers = feature_state.get("student_layers", [])
    teacher_layers = feature_state.get("teacher_layers", [])
    if not student_layers or not teacher_layers:
        return torch.zeros((), device=device)

    total = torch.zeros((), device=device)
    pair_count = 0
    for student_layer, teacher_layer in zip(student_layers, teacher_layers, strict=False):
        student_feat = feature_state["student"].get(student_layer)
        teacher_feat = feature_state["teacher"].get(teacher_layer)
        if student_feat is None or teacher_feat is None:
            continue
        if not hasattr(student_feat, "shape") or not hasattr(teacher_feat, "shape"):
            continue
        if student_feat.shape != teacher_feat.shape:
            continue
        fg_mask = build_fg_mask(
            batch=batch,
            batch_size=int(student_feat.shape[0]),
            height=int(student_feat.shape[2]),
            width=int(student_feat.shape[3]),
            device=device,
            expand_ratio=mask_expand_ratio,
        )
        diff_map = (student_feat - teacher_feat.detach()).pow(2).mean(dim=1, keepdim=True)
        total = total + weighted_spatial_mean(diff_map, fg_mask, bg_weight=bg_weight)
        pair_count += 1
    if pair_count == 0:
        return torch.zeros((), device=device)
    return total / float(pair_count)


def build_fg_mask(
    batch: dict[str, Any] | None,
    batch_size: int,
    height: int,
    width: int,
    device,
    expand_ratio: float = 0.0,
):
    import torch

    mask = torch.zeros((batch_size, 1, height, width), device=device)
    if not batch:
        return mask
    bboxes = batch.get("bboxes")
    batch_idx = batch.get("batch_idx")
    if bboxes is None or batch_idx is None:
        return mask
    if not hasattr(bboxes, "shape") or bboxes.numel() == 0:
        return mask

    batch_idx = batch_idx.view(-1).to(device=device, dtype=torch.long)
    bboxes = bboxes.to(device=device, dtype=torch.float32)
    normalized = float(bboxes.max()) <= 1.5
    for idx, box in zip(batch_idx.tolist(), bboxes, strict=False):
        if idx < 0 or idx >= batch_size:
            continue
        if normalized:
            x, y, w, h = box.tolist()
            w *= 1.0 + expand_ratio
            h *= 1.0 + expand_ratio
            x1 = max(0, int((x - w / 2.0) * width))
            y1 = max(0, int((y - h / 2.0) * height))
            x2 = min(width, int((x + w / 2.0) * width + 0.9999))
            y2 = min(height, int((y + h / 2.0) * height + 0.9999))
        else:
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2 + 0.9999))
            y2 = min(height, int(y2 + 0.9999))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[idx, 0, y1:y2, x1:x2] = 1.0
    return mask


def mask_with_background_weight(mask, bg_weight: float):
    return mask + (1.0 - mask) * max(bg_weight, 0.0)


def weighted_spatial_mean(loss_map, mask, bg_weight: float = 0.0):
    import torch

    weighted_mask = mask_with_background_weight(mask, bg_weight=bg_weight)
    weighted_loss = loss_map * weighted_mask
    denom = weighted_mask.sum()
    if float(denom) <= 0.0:
        return torch.zeros((), device=loss_map.device)
    return weighted_loss.sum() / denom
