from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.data_aug import read_manifest


@dataclass(frozen=True)
class MergeStats:
    total_rows: int
    selected_rows: int
    merged_rows: int
    skipped_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge accepted synthetic samples into a YOLO detection dataset split."
    )
    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=ROOT / "datasets" / "synthetic" / "mvtec_mock_v1" / "manifest.jsonl",
        help="Path to synthetic manifest.jsonl",
    )
    parser.add_argument(
        "--target-dataset",
        type=Path,
        default=ROOT / "datasets" / "processed" / "mvtec_detection",
        help="Path to target YOLO detection dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test"),
        help="Target split to merge into.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="If set, merge all synthetic rows regardless of accepted.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="synthetic__",
        help="Prefix added to output file names to avoid collisions.",
    )
    parser.add_argument(
        "--conflict",
        type=str,
        default="skip",
        choices=("skip", "overwrite"),
        help="Conflict handling when destination file exists.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report path. Defaults to <target-dataset>/_merge_reports/synthetic_merge_report.json",
    )
    parser.add_argument(
        "--update-data-yaml",
        action="store_true",
        help="Update or create <target-dataset>/data.yaml with normalized path/train/val/test entries.",
    )
    return parser.parse_args()


def should_select(row: dict, include_rejected: bool) -> bool:
    if row.get("source") != "synthetic":
        return False
    if include_rejected:
        return True
    return bool(row.get("accepted", False))


def replace_or_append_key(lines: list[str], key: str, value: str) -> list[str]:
    new_line = f"{key}: {value}"
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{key}:"):
            lines[index] = new_line
            return lines
    lines.append(new_line)
    return lines


def infer_names_from_manifest(target_root: Path) -> dict[int, str] | None:
    manifest_path = target_root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        return None
    return {index: str(name) for index, name in enumerate(classes)}


def infer_names_from_labels(target_root: Path) -> dict[int, str] | None:
    class_ids: set[int] = set()
    for split in ("train", "val", "test"):
        split_dir = target_root / "labels" / split
        if not split_dir.exists():
            continue
        for label_file in split_dir.glob("*.txt"):
            text = label_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_ids.add(int(parts[0]))
                except ValueError:
                    continue
    if not class_ids:
        return None
    max_id = max(class_ids)
    return {index: f"class_{index}" for index in range(max_id + 1)}


def update_or_create_data_yaml(target_root: Path) -> Path:
    data_yaml = target_root / "data.yaml"
    if data_yaml.exists():
        lines = data_yaml.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    lines = replace_or_append_key(lines, "path", str(target_root))
    lines = replace_or_append_key(lines, "train", "images/train")
    lines = replace_or_append_key(lines, "val", "images/val")
    lines = replace_or_append_key(lines, "test", "images/test")

    has_nc = any(line.strip().startswith("nc:") for line in lines)
    has_names = any(line.strip() == "names:" for line in lines)
    if not has_nc or not has_names:
        names = infer_names_from_manifest(target_root) or infer_names_from_labels(target_root) or {0: "defect"}
        if not has_nc:
            lines.append(f"nc: {len(names)}")
        if not has_names:
            lines.append("names:")
            for index in sorted(names):
                lines.append(f"  {index}: {names[index]}")

    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def main() -> None:
    args = parse_args()

    manifest_path = args.synthetic_manifest.resolve()
    target_root = args.target_dataset.resolve()
    split = args.split

    if not manifest_path.exists():
        raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)

    images_out = target_root / "images" / split
    labels_out = target_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    selected = [row for row in rows if should_select(row, include_rejected=args.include_rejected)]

    merged = 0
    skipped = 0
    merged_records: list[dict] = []

    for row in selected:
        src_image = ROOT / str(row["image_path"])
        src_label = ROOT / str(row["label_path"])
        if not src_image.exists() or not src_label.exists():
            skipped += 1
            continue

        out_image = images_out / f"{args.prefix}{src_image.name}"
        out_label = labels_out / f"{args.prefix}{src_label.name}"

        if out_image.exists() or out_label.exists():
            if args.conflict == "skip":
                skipped += 1
                continue

        shutil.copy2(src_image, out_image)
        shutil.copy2(src_label, out_label)

        merged += 1
        merged_records.append(
            {
                "prompt_id": row["prompt_id"],
                "defect_type": row["defect_type"],
                "quality_score": row["quality_score"],
                "accepted": row["accepted"],
                "image_path": str(out_image.relative_to(ROOT)),
                "label_path": str(out_label.relative_to(ROOT)),
                "source": "synthetic",
                "split": split,
            }
        )

    stats = MergeStats(
        total_rows=len(rows),
        selected_rows=len(selected),
        merged_rows=merged,
        skipped_rows=skipped,
    )

    report_path = args.report
    if report_path is None:
        report_path = target_root / "_merge_reports" / "synthetic_merge_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic_manifest": str(manifest_path),
        "target_dataset": str(target_root),
        "split": split,
        "selection_mode": "all_synthetic" if args.include_rejected else "accepted_only",
        "prefix": args.prefix,
        "conflict": args.conflict,
        "stats": asdict(stats),
        "merged_records": merged_records,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Merged {merged}/{len(selected)} selected rows into {target_root} ({split})")
    print(f"Skipped rows: {skipped}")
    print(f"Report: {report_path}")
    if args.update_data_yaml:
        data_yaml_path = update_or_create_data_yaml(target_root)
        print(f"Updated data yaml: {data_yaml_path}")


if __name__ == "__main__":
    main()
