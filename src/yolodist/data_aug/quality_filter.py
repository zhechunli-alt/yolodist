from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random

from yolodist.data_aug.manifest import read_manifest, write_manifest, ManifestRecord


@dataclass(frozen=True)
class FilterResult:
    total: int
    kept: int
    rejected: int


def apply_quality_threshold(manifest_path: Path, threshold: float) -> FilterResult:
    rows = read_manifest(manifest_path)

    output_rows: list[ManifestRecord] = []
    kept = 0
    rejected = 0
    for row in rows:
        accepted = bool(row["accepted"])
        if row["source"] == "synthetic":
            accepted = float(row["quality_score"]) >= threshold

        if accepted:
            kept += 1
        else:
            rejected += 1

        output_rows.append(
            ManifestRecord(
                image_path=row["image_path"],
                label_path=row["label_path"],
                source=row["source"],
                prompt_id=row["prompt_id"],
                defect_type=row["defect_type"],
                quality_score=float(row["quality_score"]),
                accepted=accepted,
            )
        )

    write_manifest(manifest_path, output_rows)
    return FilterResult(total=len(output_rows), kept=kept, rejected=rejected)


def export_manual_review(
    manifest_path: Path,
    output_csv: Path,
    threshold: float,
    sample_size: int,
    seed: int,
    band_width: float = 0.1,
) -> int:
    rows = read_manifest(manifest_path)
    lower = max(0.0, threshold - band_width)
    upper = min(1.0, threshold + band_width)

    candidates = [
        row
        for row in rows
        if row["source"] == "synthetic" and lower <= float(row["quality_score"]) <= upper
    ]

    rng = random.Random(seed)
    if len(candidates) > sample_size:
        candidates = rng.sample(candidates, sample_size)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prompt_id", "defect_type", "quality_score", "accepted", "image_path", "label_path"],
        )
        writer.writeheader()
        for row in sorted(candidates, key=lambda item: float(item["quality_score"])):
            writer.writerow(
                {
                    "prompt_id": row["prompt_id"],
                    "defect_type": row["defect_type"],
                    "quality_score": row["quality_score"],
                    "accepted": row["accepted"],
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],
                }
            )
    return len(candidates)
