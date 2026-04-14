from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import shutil
from typing import Iterable

from PIL import Image

from yolodist.paths import ROOT


VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class BuildConfig:
    source_dir: Path
    output_dir: Path
    class_mode: str = "binary"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    copy_images: bool = True


@dataclass
class Sample:
    image_path: Path
    split: str
    label_lines: list[str]
    category: str
    defect_type: str


def build_from_config_dict(config: dict) -> Path:
    dataset_cfg = config["dataset"]
    split_cfg = config["splits"]
    build_config = BuildConfig(
        source_dir=ROOT / dataset_cfg["source_dir"],
        output_dir=ROOT / dataset_cfg["output_dir"],
        class_mode=dataset_cfg.get("class_mode", "binary"),
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(dataset_cfg.get("seed", 42)),
        copy_images=bool(dataset_cfg.get("copy_images", True)),
    )
    return build_detection_dataset(build_config)


def build_detection_dataset(config: BuildConfig) -> Path:
    if abs((config.train_ratio + config.val_ratio + config.test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    if not config.source_dir.exists():
        raise FileNotFoundError(f"MVTec source directory not found: {config.source_dir}")

    classes = collect_class_names(config.source_dir, config.class_mode)
    samples = collect_samples(config, classes)

    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)

    for split in ("train", "val", "test"):
        (config.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for sample in samples:
        dst_image = config.output_dir / "images" / sample.split / image_name(sample)
        dst_label = config.output_dir / "labels" / sample.split / label_name(sample)
        if config.copy_images:
            shutil.copy2(sample.image_path, dst_image)
        else:
            dst_image.symlink_to(sample.image_path.resolve())
        dst_label.write_text("\n".join(sample.label_lines) + ("\n" if sample.label_lines else ""), encoding="utf-8")
        manifest.append(
            {
                "image": str(dst_image.relative_to(config.output_dir)),
                "label": str(dst_label.relative_to(config.output_dir)),
                "split": sample.split,
                "category": sample.category,
                "defect_type": sample.defect_type,
            }
        )

    write_dataset_yaml(config.output_dir, classes)
    (config.output_dir / "manifest.json").write_text(
        json.dumps({"classes": classes, "samples": manifest}, indent=2),
        encoding="utf-8",
    )
    return config.output_dir / "data.yaml"


def collect_class_names(source_dir: Path, class_mode: str) -> list[str]:
    categories = category_dirs(source_dir)
    if class_mode == "binary":
        return ["defect"]
    if class_mode == "category":
        return [path.name for path in categories]
    if class_mode == "defect_type":
        names: set[str] = set()
        for category_dir in categories:
            for defect_dir in sorted((category_dir / "test").iterdir()):
                if defect_dir.is_dir() and defect_dir.name != "good":
                    names.add(f"{category_dir.name}__{defect_dir.name}")
        return sorted(names)
    raise ValueError(f"Unsupported class mode: {class_mode}")


def collect_samples(config: BuildConfig, classes: list[str]) -> list[Sample]:
    rng = random.Random(config.seed)
    all_samples: list[Sample] = []

    for category_dir in category_dirs(config.source_dir):
        category = category_dir.name
        all_samples.extend(collect_good_training_samples(category_dir, category))
        all_samples.extend(collect_good_eval_samples(category_dir, category))
        all_samples.extend(collect_anomaly_samples(category_dir, category, config, classes, rng))

    return all_samples


def collect_good_training_samples(category_dir: Path, category: str) -> list[Sample]:
    samples: list[Sample] = []
    train_good_dir = category_dir / "train" / "good"
    for image_path in iter_images(train_good_dir):
        samples.append(Sample(image_path=image_path, split="train", label_lines=[], category=category, defect_type="good"))
    return samples


def collect_good_eval_samples(category_dir: Path, category: str) -> list[Sample]:
    test_good_dir = category_dir / "test" / "good"
    eval_good = list(iter_images(test_good_dir))
    midpoint = len(eval_good) // 2
    samples: list[Sample] = []
    for image_path in eval_good[:midpoint]:
        samples.append(Sample(image_path=image_path, split="val", label_lines=[], category=category, defect_type="good"))
    for image_path in eval_good[midpoint:]:
        samples.append(Sample(image_path=image_path, split="test", label_lines=[], category=category, defect_type="good"))
    return samples


def collect_anomaly_samples(
    category_dir: Path,
    category: str,
    config: BuildConfig,
    classes: list[str],
    rng: random.Random,
) -> list[Sample]:
    samples: list[Sample] = []
    grouped: list[tuple[Path, str]] = []
    test_dir = category_dir / "test"
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        for image_path in iter_images(defect_dir):
            grouped.append((image_path, defect_dir.name))

    rng.shuffle(grouped)
    train_cutoff = int(len(grouped) * config.train_ratio)
    val_cutoff = train_cutoff + int(len(grouped) * config.val_ratio)

    for index, (image_path, defect_type) in enumerate(grouped):
        if index < train_cutoff:
            split = "train"
        elif index < val_cutoff:
            split = "val"
        else:
            split = "test"

        mask_path = find_mask_path(category_dir / "ground_truth" / defect_type, image_path.stem)
        boxes = mask_to_yolo_boxes(mask_path)
        class_id = class_id_for(category, defect_type, config.class_mode, classes)
        label_lines = [
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            for x_center, y_center, width, height in boxes
        ]
        samples.append(
            Sample(
                image_path=image_path,
                split=split,
                label_lines=label_lines,
                category=category,
                defect_type=defect_type,
            )
        )
    return samples


def category_dirs(source_dir: Path) -> list[Path]:
    excluded = {"experiments", "runs", "weights", "processed"}
    return [
        path
        for path in sorted(source_dir.iterdir())
        if path.is_dir() and path.name not in excluded and (path / "test").exists() and (path / "ground_truth").exists()
    ]


def iter_images(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return [path for path in sorted(directory.iterdir()) if path.suffix.lower() in VALID_IMAGE_SUFFIXES]


def class_id_for(category: str, defect_type: str, class_mode: str, classes: list[str]) -> int:
    if class_mode == "binary":
        return 0
    if class_mode == "category":
        return classes.index(category)
    key = f"{category}__{defect_type}"
    return classes.index(key)


def find_mask_path(mask_dir: Path, stem: str) -> Path:
    direct = mask_dir / f"{stem}_mask.png"
    if direct.exists():
        return direct
    candidates = sorted(mask_dir.glob(f"{stem}*"))
    if not candidates:
        raise FileNotFoundError(f"No mask found for {stem} in {mask_dir}")
    return candidates[0]


def mask_to_yolo_boxes(mask_path: Path) -> list[tuple[float, float, float, float]]:
    with Image.open(mask_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        pixels = gray.load()
        visited = [[False for _ in range(width)] for _ in range(height)]
        boxes: list[tuple[int, int, int, int]] = []

        for y in range(height):
            for x in range(width):
                if visited[y][x] or pixels[x, y] == 0:
                    continue
                boxes.append(flood_fill_bbox(pixels, visited, width, height, x, y))

    return [to_yolo_box(box, width, height) for box in boxes]


def flood_fill_bbox(pixels, visited: list[list[bool]], width: int, height: int, start_x: int, start_y: int) -> tuple[int, int, int, int]:
    stack = [(start_x, start_y)]
    visited[start_y][start_x] = True
    min_x = max_x = start_x
    min_y = max_y = start_y

    while stack:
        x, y = stack.pop()
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if next_x < 0 or next_y < 0 or next_x >= width or next_y >= height:
                continue
            if visited[next_y][next_x] or pixels[next_x, next_y] == 0:
                continue
            visited[next_y][next_x] = True
            stack.append((next_x, next_y))

    return min_x, min_y, max_x, max_y


def to_yolo_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = box
    box_width = (max_x - min_x + 1) / width
    box_height = (max_y - min_y + 1) / height
    center_x = (min_x + max_x + 1) / 2 / width
    center_y = (min_y + max_y + 1) / 2 / height
    return center_x, center_y, box_width, box_height


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


def image_name(sample: Sample) -> str:
    return f"{sample.category}__{sample.defect_type}__{sample.image_path.name}"


def label_name(sample: Sample) -> str:
    return f"{sample.category}__{sample.defect_type}__{sample.image_path.stem}.txt"
