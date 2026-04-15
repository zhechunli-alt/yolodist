from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.config import load_toml
from yolodist.data_aug import apply_quality_threshold, export_manual_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply quality filter to synthetic manifest and export review list.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data_aug" / "mvtec_mock_v1.toml",
        help="Path to data augmentation TOML config.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit manifest path. Defaults to <output_dir>/manifest.jsonl from config.",
    )
    return parser.parse_args()


def resolve_output_dir(dataset_cfg: dict, config_path: Path) -> Path:
    output_dir = dataset_cfg.get("output_dir")
    if output_dir:
        return ROOT / str(output_dir)
    version = dataset_cfg.get("version", config_path.stem)
    return ROOT / "datasets" / "synthetic" / str(version)


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)

    dataset_cfg = cfg.get("dataset", {})
    quality_cfg = cfg.get("quality", {})

    output_dir = resolve_output_dir(dataset_cfg, args.config)
    manifest_path = args.manifest or (output_dir / "manifest.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    threshold = float(quality_cfg.get("filter_threshold", quality_cfg.get("auto_accept_threshold", 0.65)))
    review_sample_size = int(quality_cfg.get("manual_review_sample_size", 50))
    review_band_width = float(quality_cfg.get("manual_review_band_width", 0.1))
    seed = int(dataset_cfg.get("seed", 42))

    result = apply_quality_threshold(manifest_path=manifest_path, threshold=threshold)
    review_csv = output_dir / "manual_review.csv"
    sampled = export_manual_review(
        manifest_path=manifest_path,
        output_csv=review_csv,
        threshold=threshold,
        sample_size=review_sample_size,
        seed=seed,
        band_width=review_band_width,
    )

    print(f"Manifest filtered with threshold={threshold:.3f}")
    print(f"Total: {result.total}, kept: {result.kept}, rejected: {result.rejected}")
    print(f"Manual review list: {review_csv} ({sampled} rows)")


if __name__ == "__main__":
    main()
