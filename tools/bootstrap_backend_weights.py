from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.paths import WEIGHTS_DIR


TARGETS = [
    "best_teacher.pt",
    "best_student_plain.pt",
    "best_student_ppla.pt",
    "best_student_distill.pt",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap backend standard weights from one existing .pt file (for API smoke tests)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source .pt path, e.g. weights/yolo11n.pt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show copy plan only.",
    )
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else (ROOT / args.source)
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source.as_posix()}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using source: {source.as_posix()}")
    for name in TARGETS:
        dst = WEIGHTS_DIR / name
        if args.dry_run:
            print(f"[dry-run] {source.as_posix()} -> {dst.as_posix()}")
            continue
        shutil.copy2(source, dst)
        print(f"[ok] {source.as_posix()} -> {dst.as_posix()}")

    if not args.dry_run:
        print("")
        print("Bootstrap complete. These files are for API smoke tests only.")
        print("Replace them with real trained weights for final experiments.")


if __name__ == "__main__":
    main()

