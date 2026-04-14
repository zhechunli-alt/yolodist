from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.train.ultralytics_runner import run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modified-model training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train" / "modified_model.toml",
        help="Path to train config TOML.",
    )
    args = parser.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
