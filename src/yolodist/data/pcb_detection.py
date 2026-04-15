from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import random
import shutil
import tarfile
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageFile

from yolodist.paths import ROOT


ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PCB_DATASETS = {"deeppcb", "pku_market_pcb", "dspcbsd_plus"}
STANDARD_PCB_CLASSES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "missing_hole",
]
DSPCBSD_PLUS_CLASSES = [
    "short",
    "spur",
    "spurious_copper",
    "open",
    "mousebite",
    "hole_breakout",
    "conductor_scratch",
    "conductor_foreign_object",
    "base_material_foreign_object",
]
DEFAULT_CLASS_MAP: dict[str, list[str]] = {
    "deeppcb": STANDARD_PCB_CLASSES,
    "pku_market_pcb": STANDARD_PCB_CLASSES,
    "dspcbsd_plus": DSPCBSD_PLUS_CLASSES,
}
DEFAULT_CLASS_ALIASES: dict[str, dict[str, str]] = {
    "deeppcb": {
        "open_circuit": "open",
        "short_circuit": "short",
        "mouse_bite": "mousebite",
        "pin-hole": "missing_hole",
        "pin_hole": "missing_hole",
        "missing": "missing_hole",
        "missing-hole": "missing_hole",
        "copper": "spurious_copper",
        "spurious": "spurious_copper",
        "spurious-copper": "spurious_copper",
    },
    "pku_market_pcb": {
        "open_circuit": "open",
        "short_circuit": "short",
        "mouse_bite": "mousebite",
        "pin-hole": "missing_hole",
        "pin_hole": "missing_hole",
        "copper": "spurious_copper",
        "spurious": "spurious_copper",
        "spurious-copper": "spurious_copper",
    },
    "dspcbsd_plus": {
        "sh": "short",
        "sp": "spur",
        "sc": "spurious_copper",
        "op": "open",
        "mb": "mousebite",
        "hb": "hole_breakout",
        "cs": "conductor_scratch",
        "cfo": "conductor_foreign_object",
        "bmfo": "base_material_foreign_object",
        "mouse_bite": "mousebite",
        "spurious-copper": "spurious_copper",
    },
}
DEEPPCB_TYPE_MAP = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious_copper",
    6: "missing_hole",
}


@dataclass
class BuildConfig:
    dataset: str
    source_dir: Path
    output_dir: Path
    split_seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    copy_images: bool = True
    classes: list[str] | None = None
    alias_map: dict[str, str] = field(default_factory=dict)


@dataclass
class Sample:
    image_path: Path
    split: str
    label_lines: list[str]
    metadata: dict[str, object]


def build_from_config_dict(config: dict) -> Path:
    dataset_cfg = config["dataset"]
    build_config = BuildConfig(
        dataset=str(dataset_cfg["name"]),
        source_dir=ROOT / dataset_cfg["source_dir"],
        output_dir=ROOT / dataset_cfg["output_dir"],
        split_seed=int(dataset_cfg.get("split_seed", dataset_cfg.get("seed", 42))),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.2)),
        copy_images=bool(dataset_cfg.get("copy_images", True)),
        classes=list(dataset_cfg.get("classes", DEFAULT_CLASS_MAP[str(dataset_cfg["name"])])),
        alias_map=dict(DEFAULT_CLASS_ALIASES.get(str(dataset_cfg["name"]), {}))
        | dict(dataset_cfg.get("alias_map", {})),
    )
    return build_detection_dataset(build_config)


def build_detection_dataset(config: BuildConfig) -> Path:
    if config.dataset not in PCB_DATASETS:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    if not config.source_dir.exists():
        raise FileNotFoundError(f"PCB source directory not found: {config.source_dir}")
    if not (0.0 <= config.val_ratio < 1.0 and 0.0 <= config.test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if config.val_ratio + config.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    source_dir = maybe_extract_archives(config.source_dir)
    classes = list(config.classes or DEFAULT_CLASS_MAP[config.dataset])
    class_to_id = {name: index for index, name in enumerate(classes)}
    alias_log: list[dict[str, str]] = []

    if config.dataset == "deeppcb":
        samples = collect_deeppcb_samples(config, source_dir, class_to_id, alias_log)
    else:
        samples = collect_generic_samples(config, source_dir, class_to_id, alias_log)

    if not samples:
        raise RuntimeError(f"No PCB samples found under {source_dir}")

    if config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    for split in ("train", "val", "test"):
        (config.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (config.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for sample in samples:
        dst_image = config.output_dir / "images" / sample.split / sample.image_path.name
        dst_label = config.output_dir / "labels" / sample.split / f"{sample.image_path.stem}.txt"
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
                "num_boxes": len(sample.label_lines),
                **sample.metadata,
            }
        )

    write_dataset_yaml(config.output_dir, classes)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": config.dataset,
                "source_dir": str(source_dir),
                "classes": classes,
                "alias_log": alias_log,
                "samples": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return config.output_dir / "data.yaml"


def maybe_extract_archives(source_dir: Path) -> Path:
    # Prefer an already materialized dataset tree over re-extracting leftover archives.
    for ready_dir in (source_dir / "PCBData", source_dir / "images", source_dir / "annotations", source_dir / "pcb", source_dir / "Data_COCO"):
        if ready_dir.exists():
            return source_dir

    archives = sorted(source_dir.glob("*.zip")) + sorted(source_dir.glob("*.tar.gz")) + sorted(source_dir.glob("*.tgz"))
    if not archives:
        nested_archive = next(iter(sorted(source_dir.glob("**/*.zip"))), None)
        if nested_archive is None:
            nested_archive = next(iter(sorted(source_dir.glob("**/*.tar.gz"))), None)
        if nested_archive is None:
            nested_archive = next(iter(sorted(source_dir.glob("**/*.tgz"))), None)
        return source_dir if nested_archive is None else extract_archive(nested_archive)
    if len(archives) == 1:
        try:
            return extract_archive(archives[0])
        except (zipfile.BadZipFile, tarfile.TarError, ValueError):
            return source_dir
    return source_dir


def extract_archive(archive_path: Path) -> Path:
    extract_dir = archive_output_dir(archive_path)
    if extract_dir.exists():
        return normalize_extracted_root(extract_dir)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(extract_dir)
    elif archive_path.name.endswith(".tar.gz") or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as handle:
            handle.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return normalize_extracted_root(extract_dir)


def archive_output_dir(archive_path: Path) -> Path:
    if archive_path.name.endswith(".tar.gz"):
        return archive_path.with_name(archive_path.name[:-7])
    if archive_path.suffix == ".tgz":
        return archive_path.with_suffix("")
    return archive_path.with_suffix("")


def normalize_extracted_root(extract_dir: Path) -> Path:
    entries = [path for path in sorted(extract_dir.iterdir()) if not path.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def collect_deeppcb_samples(
    config: BuildConfig,
    source_dir: Path,
    class_to_id: dict[str, int],
    alias_log: list[dict[str, str]],
) -> list[Sample]:
    pcb_root = source_dir / "PCBData" if (source_dir / "PCBData").exists() else source_dir
    train_list = pcb_root / "trainval.txt"
    test_list = pcb_root / "test.txt"
    if not train_list.exists() or not test_list.exists():
        raise FileNotFoundError(f"DeepPCB split files not found under {pcb_root}")

    train_entries = read_deeppcb_split_entries(train_list)
    test_entries = read_deeppcb_split_entries(test_list)
    rng = random.Random(config.split_seed)
    rng.shuffle(train_entries)
    val_count = max(1, int(len(train_entries) * config.val_ratio)) if train_entries else 0
    val_images = {entry[0] for entry in train_entries[:val_count]}
    test_entry_set = set(test_entries)

    samples: list[Sample] = []
    for image_rel, label_rel in train_entries + test_entries:
        image_path = resolve_deeppcb_listed_image(pcb_root, image_rel)
        annotation_path = pcb_root / label_rel
        label_lines = read_deeppcb_annotation(annotation_path, image_path, class_to_id, alias_log)
        split = "test" if (image_rel, label_rel) in test_entry_set else ("val" if image_rel in val_images else "train")
        samples.append(
            Sample(
                image_path=image_path,
                split=split,
                label_lines=label_lines,
                metadata={
                    "dataset": "deeppcb",
                    "source_image": str(image_path.relative_to(source_dir)),
                    "source_annotation": str(annotation_path.relative_to(source_dir)),
                },
            )
        )
    return samples


def collect_generic_samples(
    config: BuildConfig,
    source_dir: Path,
    class_to_id: dict[str, int],
    alias_log: list[dict[str, str]],
) -> list[Sample]:
    if has_coco_layout(source_dir):
        return collect_coco_samples(config, source_dir, class_to_id, config.alias_map, alias_log)
    if has_yolo_split_layout(source_dir):
        return collect_yolo_split_samples(source_dir, class_to_id, config.alias_map, alias_log)
    if has_voc_layout(source_dir):
        return collect_voc_samples(source_dir, class_to_id, config.alias_map, alias_log)
    if has_flat_voc_layout(source_dir):
        return collect_flat_annotation_samples(config, source_dir, class_to_id, config.alias_map, alias_log, "voc")
    if has_flat_yolo_layout(source_dir):
        return collect_flat_annotation_samples(config, source_dir, class_to_id, config.alias_map, alias_log, "yolo")
    raise RuntimeError(
        f"Unsupported PCB dataset layout under {source_dir}. "
        "Expected YOLO split folders, Pascal VOC folders, or DeepPCB official structure."
    )


def has_yolo_split_layout(source_dir: Path) -> bool:
    return (source_dir / "images").exists() and (source_dir / "labels").exists()


def has_coco_layout(source_dir: Path) -> bool:
    return (source_dir / "annotations").exists() or (source_dir / "pcb_cocoanno").exists()


def has_voc_layout(source_dir: Path) -> bool:
    return (source_dir / "JPEGImages").exists() and (source_dir / "Annotations").exists()


def has_flat_voc_layout(source_dir: Path) -> bool:
    xml_count = len(list(source_dir.glob("**/*.xml")))
    image_count = len(list_images(source_dir))
    return xml_count > 0 and image_count > 0


def has_flat_yolo_layout(source_dir: Path) -> bool:
    txt_count = len(list(source_dir.glob("**/*.txt")))
    image_count = len(list_images(source_dir))
    return txt_count > 0 and image_count > 0


def collect_yolo_split_samples(
    source_dir: Path,
    class_to_id: dict[str, int],
    alias_map: dict[str, str],
    alias_log: list[dict[str, str]],
) -> list[Sample]:
    samples: list[Sample] = []
    for split in ("train", "val", "test"):
        image_dir = source_dir / "images" / split
        label_dir = source_dir / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in list_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            label_lines = read_yolo_annotation(label_path, class_to_id, alias_map, alias_log)
            samples.append(
                Sample(
                    image_path=image_path,
                    split=split,
                    label_lines=label_lines,
                    metadata={
                        "dataset": source_dir.name,
                        "source_image": str(image_path.relative_to(source_dir)),
                        "source_annotation": str(label_path.relative_to(source_dir)),
                    },
                )
            )
    return samples


def collect_coco_samples(
    config: BuildConfig,
    source_dir: Path,
    class_to_id: dict[str, int],
    alias_map: dict[str, str],
    alias_log: list[dict[str, str]],
) -> list[Sample]:
    annotations_dir = source_dir / "annotations"
    if not annotations_dir.exists():
        annotations_dir = source_dir / "pcb_cocoanno"
    default_image_dir = source_dir / "images"
    split_to_json = {
        "train": annotations_dir / "instances_train2017.json",
        "val": annotations_dir / "instances_val2017.json",
        "test": annotations_dir / "instances_test2017.json",
    }
    if not split_to_json["train"].exists():
        split_to_json["train"] = annotations_dir / "train.json"
    if not split_to_json["val"].exists():
        split_to_json["val"] = annotations_dir / "val.json"
    if not split_to_json["test"].exists():
        split_to_json["test"] = annotations_dir / "test.json"
    split_to_image_dir = {
        "train": source_dir / "train2017" if (source_dir / "train2017").exists() else default_image_dir,
        "val": source_dir / "val2017" if (source_dir / "val2017").exists() else default_image_dir,
        "test": source_dir / "test2017" if (source_dir / "test2017").exists() else default_image_dir,
    }
    samples: list[Sample] = []
    for split, annotation_path in split_to_json.items():
        image_dir = split_to_image_dir[split]
        if not annotation_path.exists() or not image_dir.exists():
            continue
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        category_id_to_name = {
            int(item["id"]): normalize_class_name(str(item["name"]), alias_map, alias_log, annotation_path.name)
            for item in payload.get("categories", [])
        }
        image_id_to_meta = {int(item["id"]): item for item in payload.get("images", [])}
        image_to_boxes: dict[int, list[str]] = {image_id: [] for image_id in image_id_to_meta}
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            category_name = category_id_to_name.get(int(annotation["category_id"]))
            if category_name is None or category_name not in class_to_id:
                continue
            image_meta = image_id_to_meta.get(image_id)
            if image_meta is None:
                continue
            x, y, w, h = [float(value) for value in annotation["bbox"]]
            width = int(image_meta["width"])
            height = int(image_meta["height"])
            image_to_boxes[image_id].append(
                box_xyxy_to_yolo(class_to_id[category_name], x, y, x + w, y + h, width, height)
            )

        val_test_ids: set[int] = set()
        if split == "val" and not split_to_json["test"].exists():
            image_ids = sorted(image_id_to_meta)
            holdout_ratio = config.val_ratio + config.test_ratio
            test_fraction = 0.5 if holdout_ratio <= 0 else (config.test_ratio / holdout_ratio)
            split_point = int(len(image_ids) * (1.0 - test_fraction))
            val_test_ids = set(image_ids[split_point:])

        for image_id, image_meta in image_id_to_meta.items():
            image_path = image_dir / str(image_meta["file_name"])
            if not image_path.exists():
                continue
            current_split = split
            if split == "val" and val_test_ids:
                current_split = "test" if image_id in val_test_ids else "val"
            samples.append(
                Sample(
                    image_path=image_path,
                    split=current_split,
                    label_lines=image_to_boxes.get(image_id, []),
                    metadata={
                        "dataset": source_dir.name,
                        "source_image": str(image_path.relative_to(source_dir)),
                        "source_annotation": str(annotation_path.relative_to(source_dir)),
                    },
                )
            )
    return samples


def collect_voc_samples(
    source_dir: Path,
    class_to_id: dict[str, int],
    alias_map: dict[str, str],
    alias_log: list[dict[str, str]],
) -> list[Sample]:
    samples: list[Sample] = []
    split_files = resolve_voc_split_files(source_dir)
    for split, file_ids in split_files.items():
        for file_id in file_ids:
            image_path = resolve_voc_image(source_dir / "JPEGImages", file_id)
            annotation_path = source_dir / "Annotations" / f"{file_id}.xml"
            label_lines = read_voc_annotation(annotation_path, class_to_id, alias_map, alias_log)
            samples.append(
                Sample(
                    image_path=image_path,
                    split=split,
                    label_lines=label_lines,
                    metadata={
                        "dataset": source_dir.name,
                        "source_image": str(image_path.relative_to(source_dir)),
                        "source_annotation": str(annotation_path.relative_to(source_dir)),
                    },
                )
            )
    return samples


def collect_flat_annotation_samples(
    config: BuildConfig,
    source_dir: Path,
    class_to_id: dict[str, int],
    alias_map: dict[str, str],
    alias_log: list[dict[str, str]],
    annotation_format: str,
) -> list[Sample]:
    images = list_images(source_dir)
    image_to_label: list[tuple[Path, Path]] = []
    for image_path in images:
        suffix = ".xml" if annotation_format == "voc" else ".txt"
        label_path = next(iter(sorted(source_dir.glob(f"**/{image_path.stem}{suffix}"))), None)
        if label_path is None:
            continue
        image_to_label.append((image_path, label_path))

    rng = random.Random(config.split_seed)
    rng.shuffle(image_to_label)
    train_cutoff = int(len(image_to_label) * (1.0 - config.val_ratio - config.test_ratio))
    val_cutoff = int(len(image_to_label) * (1.0 - config.test_ratio))

    samples: list[Sample] = []
    for index, (image_path, label_path) in enumerate(image_to_label):
        split = "train" if index < train_cutoff else ("val" if index < val_cutoff else "test")
        if annotation_format == "voc":
            label_lines = read_voc_annotation(label_path, class_to_id, alias_map, alias_log)
        else:
            label_lines = read_yolo_annotation(label_path, class_to_id, alias_map, alias_log)
        samples.append(
            Sample(
                image_path=image_path,
                split=split,
                label_lines=label_lines,
                metadata={
                    "dataset": source_dir.name,
                    "source_image": str(image_path.relative_to(source_dir)),
                    "source_annotation": str(label_path.relative_to(source_dir)),
                },
            )
        )
    return samples


def resolve_voc_split_files(source_dir: Path) -> dict[str, list[str]]:
    image_sets_dir = source_dir / "ImageSets" / "Main"
    if image_sets_dir.exists():
        split_files = {
            "train": image_sets_dir / "train.txt",
            "val": image_sets_dir / "val.txt",
            "test": image_sets_dir / "test.txt",
        }
        available = {split: read_id_list(path) for split, path in split_files.items() if path.exists()}
        if available:
            if "train" in available and "val" in available and "test" in available:
                return available
            if "train" in available and "val" not in available:
                ids = available["train"]
                split_at = max(1, int(len(ids) * 0.9))
                return {"train": ids[:split_at], "val": ids[split_at:], "test": available.get("test", [])}
            return available
    xml_ids = [path.stem for path in sorted((source_dir / "Annotations").glob("*.xml"))]
    train_cutoff = int(len(xml_ids) * 0.7)
    val_cutoff = int(len(xml_ids) * 0.85)
    return {
        "train": xml_ids[:train_cutoff],
        "val": xml_ids[train_cutoff:val_cutoff],
        "test": xml_ids[val_cutoff:],
    }


def resolve_voc_image(image_dir: Path, file_id: str) -> Path:
    candidates = [path for path in sorted(image_dir.glob(f"{file_id}.*")) if path.suffix.lower() in VALID_IMAGE_SUFFIXES]
    if not candidates:
        raise FileNotFoundError(f"No image found for VOC id {file_id} in {image_dir}")
    return candidates[0]


def read_id_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_deeppcb_split_entries(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        image_rel, label_rel = line.split()
        entries.append((image_rel, label_rel))
    return entries


def resolve_deeppcb_listed_image(pcb_root: Path, image_rel: str) -> Path:
    direct = pcb_root / image_rel
    if direct.exists():
        return direct
    candidate = direct.with_name(f"{direct.stem}_test{direct.suffix}")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"DeepPCB image listed in split file was not found: {image_rel}")


def read_deeppcb_annotation(
    annotation_path: Path,
    image_path: Path,
    class_to_id: dict[str, int],
    alias_log: list[dict[str, str]],
) -> list[str]:
    label_lines: list[str] = []
    with Image.open(image_path) as image:
        width, height = image.size
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "," in line:
            parts = [part.strip() for part in line.split(",") if part.strip()]
        else:
            parts = [part.strip() for part in line.split() if part.strip()]
        if len(parts) != 5:
            raise ValueError(f"Unexpected DeepPCB annotation format in {annotation_path}: {line}")
        xmin, ymin, xmax, ymax = [float(value) for value in parts[:4]]
        cls_name = normalize_class_name(DEEPPCB_TYPE_MAP[int(parts[4])], {}, alias_log, "deeppcb")
        label_lines.append(box_xyxy_to_yolo(class_to_id[cls_name], xmin, ymin, xmax, ymax, width, height))
    return label_lines


def read_voc_annotation(
    annotation_path: Path,
    class_to_id: dict[str, int],
    alias_map: dict[str, str],
    alias_log: list[dict[str, str]],
) -> list[str]:
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing VOC <size> in {annotation_path}")
    width = int(float(size.findtext("width", default="0")))
    height = int(float(size.findtext("height", default="0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {annotation_path}")

    label_lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        mapped_name = normalize_class_name(name, alias_map, alias_log, annotation_path.name)
        if mapped_name not in class_to_id:
            raise KeyError(f"Unknown class `{name}` in {annotation_path}")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))
        label_lines.append(box_xyxy_to_yolo(class_to_id[mapped_name], xmin, ymin, xmax, ymax, width, height))
    return label_lines


def read_yolo_annotation(
    annotation_path: Path,
    class_to_id: dict[str, int],
    alias_map: dict[str, str],
    alias_log: list[dict[str, str]],
) -> list[str]:
    out: list[str] = []
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 columns in YOLO annotation: {annotation_path}")
        cls_token = parts[0]
        if cls_token.lstrip("-").isdigit():
            cls_id = int(cls_token)
            if cls_id < 0 or cls_id >= len(class_to_id):
                raise ValueError(f"YOLO class id out of range in {annotation_path}: {cls_id}")
        else:
            cls_name = normalize_class_name(cls_token, alias_map, alias_log, annotation_path.name)
            cls_id = class_to_id[cls_name]
        out.append(f"{cls_id} {float(parts[1]):.6f} {float(parts[2]):.6f} {float(parts[3]):.6f} {float(parts[4]):.6f}")
    return out


def normalize_class_name(
    raw_name: str,
    alias_map: dict[str, str],
    alias_log: list[dict[str, str]],
    source: str,
) -> str:
    key = raw_name.strip().lower().replace(" ", "_")
    mapped = alias_map.get(key, key)
    if mapped != key:
        alias_log.append({"source": source, "raw": key, "mapped": mapped})
    return mapped


def box_xyxy_to_yolo(cls_id: int, xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> str:
    xmin = min(max(xmin, 0.0), width)
    ymin = min(max(ymin, 0.0), height)
    xmax = min(max(xmax, 0.0), width)
    ymax = min(max(ymax, 0.0), height)
    box_w = max(xmax - xmin, 1.0)
    box_h = max(ymax - ymin, 1.0)
    x_center = xmin + box_w / 2.0
    y_center = ymin + box_h / 2.0
    return (
        f"{cls_id} "
        f"{x_center / width:.6f} "
        f"{y_center / height:.6f} "
        f"{box_w / width:.6f} "
        f"{box_h / height:.6f}"
    )


def list_images(root: Path) -> list[Path]:
    return [path for path in sorted(root.glob("**/*")) if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES]


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend([f"  {index}: {name}" for index, name in enumerate(classes)])
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
