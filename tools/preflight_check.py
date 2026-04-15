from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

from PIL import Image, ImageFile


ROOT = Path(__file__).resolve().parents[1]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_module(name: str) -> tuple[bool, str]:
    found = importlib.util.find_spec(name) is not None
    return found, f"python module `{name}` {'OK' if found else 'MISSING'}"


def check_path(path: Path, label: str) -> tuple[bool, str]:
    exists = path.exists()
    return exists, f"{label} `{path}` {'OK' if exists else 'MISSING'}"


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def add_train_checks(checks: list[tuple[bool, str]], config_path: Path) -> None:
    config = load_toml(config_path)
    if "train" in config:
        section_name = "train"
        model_key = "model"
    elif "distill" in config:
        section_name = "distill"
        model_key = "student_model"
    else:
        raise KeyError(f"Unsupported config structure in {config_path}")

    section = config[section_name]
    checks.append(check_path(config_path, f"{section_name} config"))
    checks.append(check_path(ROOT / section[model_key], model_key))

    if "data" in section:
        checks.append(check_path(ROOT / section["data"], "dataset yaml"))
    if "prepared_data" in section:
        checks.append(check_path(ROOT / section["prepared_data"], "prepared dataset yaml"))
    if "teacher_weights" in section:
        checks.append(check_path(ROOT / section["teacher_weights"], "teacher weights"))
    if "pretrained" in section:
        checks.append(check_path(ROOT / section["pretrained"], "pretrained checkpoint"))
    if "student_pretrained" in section:
        checks.append(check_path(ROOT / section["student_pretrained"], "student pretrained checkpoint"))


def read_dataset_yaml(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8")
    result: dict[str, object] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "names:":
            in_names = True
            continue
        if in_names and raw_line.startswith("  "):
            key, value = line.split(":", maxsplit=1)
            names[int(key.strip())] = value.strip()
            continue
        in_names = False
        if ":" in line:
            key, value = line.split(":", maxsplit=1)
            result[key.strip()] = value.strip()
    if names:
        result["names"] = names
    return result


def analyze_processed_dataset(data_yaml: Path) -> list[tuple[bool, str]]:
    parsed = read_dataset_yaml(data_yaml)
    dataset_root = Path(str(parsed["path"]))
    names = parsed.get("names", {})
    num_classes = len(names) if isinstance(names, dict) else 0
    checks: list[tuple[bool, str]] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_paths = [path for path in sorted(image_dir.glob("*")) if path.is_file()]
        label_paths = [path for path in sorted(label_dir.glob("*.txt")) if path.is_file()]
        checks.append((bool(image_paths), f"{split} images `{image_dir}` count={len(image_paths)}"))
        checks.append((len(image_paths) == len(label_paths), f"{split} label parity images={len(image_paths)} labels={len(label_paths)}"))
        empty_labels = 0
        invalid_labels = 0
        corrupt_images = 0
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                invalid_labels += 1
                continue
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    invalid_labels += 1
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    invalid_labels += 1
                    continue
                if cls_id < 0 or cls_id >= num_classes:
                    invalid_labels += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                corrupt_images += 1
        ratio = (empty_labels / len(image_paths)) if image_paths else 0.0
        checks.append((corrupt_images == 0, f"{split} corrupt image count={corrupt_images}"))
        checks.append((invalid_labels == 0, f"{split} invalid/class-oob label count={invalid_labels}"))
        checks.append((True, f"{split} empty-label ratio={ratio:.3f}"))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Check training dependencies and required files.")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=ROOT / "configs" / "train" / "modified_model.toml",
        help="Path to a train or distill TOML config to validate.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "datasets" / "mvtec",
        help="Dataset root directory to validate.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=None,
        help="Optional processed dataset data.yaml or data config TOML to validate in detail.",
    )
    args = parser.parse_args()

    checks: list[tuple[bool, str]] = []
    for module in ("ultralytics", "torch", "yaml", "PIL"):
        checks.append(check_module(module))

    checks.append(check_path(ROOT / "weights" / "yolo11n.pt", "weights"))
    checks.append(check_path(args.dataset_dir, "dataset dir"))
    checks.append(check_path(ROOT / "configs" / "models" / "yolo11_student.yaml", "model config"))
    add_train_checks(checks, args.train_config)
    if args.data_config is not None:
        resolved = args.data_config
        checks.append(check_path(resolved, "data config"))
        if resolved.suffix == ".yaml":
            checks.extend(analyze_processed_dataset(resolved))
        elif resolved.suffix == ".toml":
            config = load_toml(resolved)
            dataset_section = config.get("dataset", {})
            if "source_dir" in dataset_section:
                checks.append(check_path(ROOT / dataset_section["source_dir"], "raw dataset dir"))
            if "output_dir" in dataset_section:
                output_dir = ROOT / dataset_section["output_dir"]
                checks.append(check_path(output_dir, "processed dataset dir"))
                data_yaml = output_dir / "data.yaml"
                if data_yaml.exists():
                    checks.extend(analyze_processed_dataset(data_yaml))

    all_ok = True
    for ok, msg in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {msg}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nPreflight failed. Fix missing items before training.")
        sys.exit(1)
    print("\nPreflight passed. You can start server-side training.")


if __name__ == "__main__":
    main()
