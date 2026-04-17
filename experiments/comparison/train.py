from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.compare.runner import run_comparison_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparison model training.")
    parser.add_argument("--config", type=Path, required=True, help="Path to comparison train config TOML.")
    args = parser.parse_args()
    run_comparison_train(args.config)


if __name__ == "__main__":
    main()
