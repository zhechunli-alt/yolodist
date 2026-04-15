from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import shutil
import xml.etree.ElementTree as ET

from yolodist.paths import ROOT


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


@dataclass
class BuildConfig:
    source_dir: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    copy_images: bool = True
    image_dir_name: str | None = None
    annotation_dir_name: str | None = None
    annotation_format: str = "auto"
    classes: list[str] | None = None


def build_from_config_dict(config: dict) -> Path:
    dataset_cfg = config["dataset"]
    split_cfg = config["splits"]
    build_config = BuildConfig(
        source_dir=ROOT / dataset_cfg["source_dir"],
        output_dir=ROOT / dataset_cfg["output_dir"],
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(dataset_cfg.get("seed", 42)),
        copy_images=bool(dataset_cfg.get("copy_images", True)),
        image_dir_name=dataset_cfg.get("image_dir_name"),
        annotation_dir_name=dataset_cfg.get("annotation_dir_name"),
        annotation_format=str(dataset_cfg.get("annotation_format", "auto")),
        classes=list(dataset_cfg.get("classes", DEFAULT_CLASSES)),
    )
    return build_detection_dataset(build_config)


def build_detection_dataset(config: BuildConfig) -> Path:
    if abs((config.train_ratio + config.val_ratio + config.test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    if not config.source_dir.exists():
        raise FileNotFoundError(f"NEU-DET source directory not found: {config.source_dir}")

    image_dir = resolve_data_dir(
        root=config.source_dir,
        explicit_name=config.image_dir_name,
        candidates=("IMAGES", "images", "JPEGImages"),
    )
    annotation_dir = resolve_data_dir(
        root=config.source_dir,
        explicit_name=config.annotation_dir_name,
        candidates=("ANNOTATIONS", "annotations", "Annotations"),
    )
    annotation_format = detect_annotation_format(annotation_dir, config.annotation_format)
    classes = list(config.classes or DEFAULT_CLASSES)
    class_to_id = {name: index for index, name in enumerate(classes)}

    samples = collect_samples(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        annotation_format=annotation_format,
        class_to_id=class_to_id,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    if not samples:
        raise RuntimeError(
            f"No annotated samples found under {image_dir} with labels from {annotation_dir}. "
            "Check the extracted NEU-DET layout and config."
        )

    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)

    for split in ("train", "val", "test"):
        (config.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str | int]] = []
    for sample in samples:
        split = sample["split"]
        image_path = sample["image_path"]
        label_lines = sample["label_lines"]
        dst_image = config.output_dir / "images" / split / image_path.name
        dst_label = config.output_dir / "labels" / split / f"{image_path.stem}.txt"
        if config.copy_images:
            shutil.copy2(image_path, dst_image)
        else:
            dst_image.symlink_to(image_path.resolve())
        dst_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        manifest.append(
            {
                "image": str(dst_image.relative_to(config.output_dir)),
                "label": str(dst_label.relative_to(config.output_dir)),
                "split": split,
                "source_image": str(image_path.relative_to(config.source_dir)),
                "num_boxes": len(label_lines),
            }
        )

    write_dataset_yaml(config.output_dir, classes)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(config.source_dir),
                "annotation_format": annotation_format,
                "classes": classes,
                "samples": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return config.output_dir / "data.yaml"


def resolve_data_dir(root: Path, explicit_name: str | None, candidates: tuple[str, ...]) -> Path:
    if explicit_name:
        path = root / explicit_name
        if not path.exists():
            raise FileNotFoundError(f"Configured directory not found: {path}")
        return path
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of {candidates} under {root}")


def detect_annotation_format(annotation_dir: Path, preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if any(annotation_dir.glob("*.xml")):
        return "voc_xml"
    if any(annotation_dir.glob("*.txt")):
        return "yolo_txt"
    raise RuntimeError(f"Unsupported annotation format under {annotation_dir}")


def collect_samples(
    image_dir: Path,
    annotation_dir: Path,
    annotation_format: str,
    class_to_id: dict[str, int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    samples: list[dict[str, object]] = []
    image_paths = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in VALID_IMAGE_SUFFIXES]

    for image_path in image_paths:
        annotation_path = resolve_annotation_path(annotation_dir, image_path, annotation_format)
        if annotation_path is None:
            continue
        label_lines = read_annotation(annotation_path, image_path, annotation_format, class_to_id)
        if not label_lines:
            continue
        samples.append(
            {
                "image_path": image_path,
                "label_lines": label_lines,
            }
        )

    rng.shuffle(samples)
    train_cutoff = int(len(samples) * train_ratio)
    val_cutoff = train_cutoff + int(len(samples) * val_ratio)
    for index, sample in enumerate(samples):
        if index < train_cutoff:
            sample["split"] = "train"
        elif index < val_cutoff:
            sample["split"] = "val"
        else:
            sample["split"] = "test"
    return samples


def resolve_annotation_path(annotation_dir: Path, image_path: Path, annotation_format: str) -> Path | None:
    suffix = ".xml" if annotation_format == "voc_xml" else ".txt"
    direct = annotation_dir / f"{image_path.stem}{suffix}"
    if direct.exists():
        return direct
    candidates = sorted(annotation_dir.glob(f"{image_path.stem}*{suffix}"))
    if candidates:
        return candidates[0]
    return None


def read_annotation(
    annotation_path: Path,
    image_path: Path,
    annotation_format: str,
    class_to_id: dict[str, int],
) -> list[str]:
    if annotation_format == "voc_xml":
        return read_voc_xml(annotation_path, class_to_id)
    if annotation_format == "yolo_txt":
        return read_yolo_txt(annotation_path, image_path, class_to_id)
    raise ValueError(f"Unsupported annotation_format: {annotation_format}")


def read_voc_xml(annotation_path: Path, class_to_id: dict[str, int]) -> list[str]:
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {annotation_path}")
    width = int(float(size.findtext("width", default="0")))
    height = int(float(size.findtext("height", default="0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {annotation_path}")

    label_lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name not in class_to_id:
            raise KeyError(f"Unknown class `{name}` in {annotation_path}")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))
        label_lines.append(box_xyxy_to_yolo(class_to_id[name], xmin, ymin, xmax, ymax, width, height))
    return label_lines


def read_yolo_txt(annotation_path: Path, image_path: Path, class_to_id: dict[str, int]) -> list[str]:
    lines = [line.strip() for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: list[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 columns in YOLO txt annotation: {annotation_path}")
        cls_key = parts[0]
        cls_id = int(cls_key) if cls_key.isdigit() else class_to_id[cls_key]
        out.append(f"{cls_id} {float(parts[1]):.6f} {float(parts[2]):.6f} {float(parts[3]):.6f} {float(parts[4]):.6f}")
    return out


def box_xyxy_to_yolo(cls_id: int, xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> str:
    box_width = max(0.0, xmax - xmin) / width
    box_height = max(0.0, ymax - ymin) / height
    center_x = (xmin + xmax) / 2.0 / width
    center_y = (ymin + ymax) / 2.0 / height
    return f"{cls_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> None:
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend([f"  {index}: {name}" for index, name in enumerate(classes)])
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
