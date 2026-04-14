from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.reporting.summary import summarize_run, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an Ultralytics run directory.")
    parser.add_argument("run_dir", type=Path, help="Path to a training or evaluation run directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output markdown path. Defaults to <run_dir>/summary.md",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = summarize_run(run_dir)
    markdown = to_markdown(summary)
    output = args.output or (run_dir / "summary.md")
    output.write_text(markdown, encoding="utf-8")
    print(f"Summary written to: {output}")


if __name__ == "__main__":
    main()
