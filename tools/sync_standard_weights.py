from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.paths import RUNS_DIR, WEIGHTS_DIR


def _find_latest_best(run_root: Path, include_keywords: List[str] | None = None) -> Path | None:
    if not run_root.exists():
        return None
    include_keywords = [k.lower() for k in (include_keywords or [])]
    candidates_raw = list(run_root.glob("**/weights/best.pt"))
    if include_keywords:
        filtered: List[Path] = []
        for p in candidates_raw:
            run_name = p.parents[1].name.lower()
            if all(k in run_name for k in include_keywords):
                filtered.append(p)
        candidates_raw = filtered
    candidates = sorted(candidates_raw, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_sources() -> Dict[str, Path | None]:
    plain_src = _find_latest_best(RUNS_DIR / "modified", include_keywords=["plain"])
    ppla_src = _find_latest_best(RUNS_DIR / "modified", include_keywords=["ppla"])
    # Fallback to latest modified run when keyword-specific runs are not available.
    fallback_modified = _find_latest_best(RUNS_DIR / "modified")
    teacher_legacy = WEIGHTS_DIR / "teacher.pt"
    student_legacy = WEIGHTS_DIR / "student.pt"
    return {
        "best_teacher.pt": _find_latest_best(RUNS_DIR / "baseline")
        or (teacher_legacy if teacher_legacy.exists() else None),
        "best_student_plain.pt": plain_src
        or fallback_modified
        or (student_legacy if student_legacy.exists() else None),
        "best_student_ppla.pt": ppla_src
        or fallback_modified
        or (student_legacy if student_legacy.exists() else None),
        "best_student_distill.pt": _find_latest_best(RUNS_DIR / "distill"),
    }


def sync(dry_run: bool = False) -> Tuple[List[str], List[str]]:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    skipped: List[str] = []

    for target_name, src in _resolve_sources().items():
        target_path = WEIGHTS_DIR / target_name
        if src is None:
            skipped.append(f"{target_name}: missing source run weights")
            continue
        if dry_run:
            copied.append(f"[dry-run] {src.as_posix()} -> {target_path.as_posix()}")
            continue
        shutil.copy2(src, target_path)
        copied.append(f"{src.as_posix()} -> {target_path.as_posix()}")

    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync latest run weights into backend standard names."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned copies without writing files.",
    )
    args = parser.parse_args()

    copied, skipped = sync(dry_run=args.dry_run)
    print(f"Project root: {ROOT.as_posix()}")
    print(f"Weights dir: {WEIGHTS_DIR.as_posix()}")
    print("")
    print("Copied:")
    if copied:
        for line in copied:
            print(f"- {line}")
    else:
        print("- (none)")
    print("")
    print("Skipped:")
    if skipped:
        for line in skipped:
            print(f"- {line}")
    else:
        print("- (none)")


if __name__ == "__main__":
    main()
