#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil

from PIL import Image
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepPCB YOLO-format dataset to COCO layout for YOLOX.")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("datasets/processed/deeppcb_detection/data.yaml"),
        help="YOLO data.yaml path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/processed/deeppcb_coco"),
        help="Output COCO-style dataset directory.",
    )
    args = parser.parse_args()
    convert_to_coco(args.data_yaml.resolve(), args.out.resolve())


def convert_to_coco(data_yaml: Path, out_dir: Path) -> None:
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(spec.get("path") or data_yaml.parent)
    names = spec["names"]
    categories = [{"id": int(idx) + 1, "name": name} for idx, name in names.items()]

    mapping = {"train": "train2017", "val": "val2017", "test": "test2017"}
    annotations_dir = out_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for split, coco_name in mapping.items():
        image_src_dir = root / str(spec[split])
        image_dst_dir = out_dir / coco_name
        image_dst_dir.mkdir(parents=True, exist_ok=True)
        label_dir = root / "labels" / split

        images = []
        annotations = []
        ann_id = 1
        for img_id, image_path in enumerate(sorted(image_src_dir.glob("*")), start=1):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target_path = image_dst_dir / image_path.name
            if not target_path.exists():
                try:
                    target_path.symlink_to(image_path.resolve())
                except FileExistsError:
                    pass
                except OSError:
                    shutil.copy2(image_path, target_path)

            width, height = Image.open(image_path).size
            images.append(
                {
                    "id": img_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                cls_id, cx, cy, bw, bh = line.split()
                cx_f = float(cx) * width
                cy_f = float(cy) * height
                bw_f = float(bw) * width
                bh_f = float(bh) * height
                x = cx_f - bw_f / 2.0
                y = cy_f - bh_f / 2.0
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls_id) + 1,
                        "bbox": [x, y, bw_f, bh_f],
                        "area": bw_f * bh_f,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        payload = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        (annotations_dir / f"instances_{coco_name}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
