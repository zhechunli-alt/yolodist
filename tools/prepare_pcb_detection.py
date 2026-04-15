from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.data.pcb_detection import BuildConfig, DEFAULT_CLASS_ALIASES, DEFAULT_CLASS_MAP, build_detection_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PCB datasets to Ultralytics detection format.")
    parser.add_argument("--dataset", required=True, choices=("deeppcb", "pku_market_pcb", "dspcbsd_plus"))
    parser.add_argument("--src", required=True, type=Path, help="Raw dataset directory or archive directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output processed dataset directory.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used for random split fallback.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for datasets without official val split.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test ratio for datasets without official test split.")
    parser.add_argument("--symlink-images", action="store_true", help="Symlink images instead of copying.")
    args = parser.parse_args()

    output_yaml = build_detection_dataset(
        BuildConfig(
            dataset=args.dataset,
            source_dir=args.src.resolve(),
            output_dir=args.out.resolve(),
            split_seed=args.split_seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            copy_images=not args.symlink_images,
            classes=list(DEFAULT_CLASS_MAP[args.dataset]),
            alias_map=dict(DEFAULT_CLASS_ALIASES.get(args.dataset, {})),
        )
    )
    print(f"Prepared dataset yaml: {output_yaml}")


if __name__ == "__main__":
    main()
