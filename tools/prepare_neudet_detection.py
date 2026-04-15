from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.config import load_toml
from yolodist.data.neudet_detection import build_from_config_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NEU-DET to a YOLO detection dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data" / "neudet_detection.toml",
        help="Path to dataset conversion TOML config.",
    )
    args = parser.parse_args()
    output_yaml = build_from_config_dict(load_toml(args.config))
    print(f"Prepared dataset yaml: {output_yaml}")


if __name__ == "__main__":
    main()
