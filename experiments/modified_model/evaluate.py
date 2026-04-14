from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.eval.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modified-model evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eval" / "modified_model.toml",
        help="Path to eval config TOML.",
    )
    args = parser.parse_args()
    summary_path = run_eval(args.config)
    print(f"Evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
