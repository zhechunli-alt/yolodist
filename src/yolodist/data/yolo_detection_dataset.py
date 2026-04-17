from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    label_path: Path


class YoloDetectionDataset(Dataset):
    def __init__(self, data_yaml: Path, split: str, imgsz: int) -> None:
        spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        root = Path(spec.get("path") or data_yaml.parent)
        image_dir = root / str(spec[split])
        label_dir = root / "labels" / split
        self.samples = [
            DetectionSample(image_path=path, label_path=label_dir / f"{path.stem}.txt")
            for path in sorted(image_dir.glob("*"))
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        self.imgsz = int(imgsz)
        self.num_classes = int(spec["nc"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        orig_w, orig_h = image.size
        resized = image.resize((self.imgsz, self.imgsz))
        image_tensor = torch.from_numpy(__import__("numpy").array(resized)).permute(2, 0, 1).float() / 255.0

        boxes = []
        labels = []
        if sample.label_path.exists():
            for raw_line in sample.label_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                cls_id, cx, cy, bw, bh = line.split()
                cx_f = float(cx) * orig_w
                cy_f = float(cy) * orig_h
                bw_f = float(bw) * orig_w
                bh_f = float(bh) * orig_h
                x1 = (cx_f - bw_f / 2.0) * self.imgsz / orig_w
                y1 = (cy_f - bh_f / 2.0) * self.imgsz / orig_h
                x2 = (cx_f + bw_f / 2.0) * self.imgsz / orig_w
                y2 = (cy_f + bh_f / 2.0) * self.imgsz / orig_h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id) + 1)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
