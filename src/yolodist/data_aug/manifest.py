from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


MANIFEST_FIELDS = (
    "image_path",
    "label_path",
    "source",
    "prompt_id",
    "defect_type",
    "quality_score",
    "accepted",
)


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    label_path: str
    source: str
    prompt_id: str
    defect_type: str
    quality_score: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "label_path": self.label_path,
            "source": self.source,
            "prompt_id": self.prompt_id,
            "defect_type": self.defect_type,
            "quality_score": round(float(self.quality_score), 6),
            "accepted": bool(self.accepted),
        }


def write_manifest(path: Path, records: Iterable[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            validate_manifest_row(row, line_number)
            rows.append(row)
    return rows


def validate_manifest_row(row: dict, line_number: int | None = None) -> None:
    missing = [field for field in MANIFEST_FIELDS if field not in row]
    if missing:
        prefix = f"line {line_number}: " if line_number is not None else ""
        raise ValueError(f"{prefix}manifest row missing fields: {missing}")
