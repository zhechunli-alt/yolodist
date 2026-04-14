from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.distill.pseudo_label import run_distillation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distillation training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train" / "distillation.toml",
        help="Path to distillation config TOML.",
    )
    args = parser.parse_args()
    run_distillation(args.config)


if __name__ == "__main__":
    main()
