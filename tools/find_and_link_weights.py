from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.paths import WEIGHTS_DIR


def find_pt_files(search_roots: list[Path], name_hint: str) -> list[Path]:
    hits: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.pt"):
            if name_hint.lower() in path.name.lower():
                hits.append(path)
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find weight files by name hints and copy to standard backend names."
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=["/Users/lizhechun/Desktop", "/Users/lizhechun/Downloads"],
        help="Search root directory; can be provided multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations.",
    )
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    roots = [Path(x).expanduser().resolve() for x in args.search_root]

    mapping = {
        "best_teacher.pt": ["teacher", "baseline", "best"],
        "best_student_plain.pt": ["plain", "student", "best"],
        "best_student_ppla.pt": ["ppla", "student", "best"],
        "best_student_distill.pt": ["distill", "student", "best"],
    }

    print("Search roots:")
    for root in roots:
        print(f"- {root.as_posix()}")

    for target, hints in mapping.items():
        candidates: list[Path] = []
        for hint in hints:
            candidates.extend(find_pt_files(roots, hint))
        # Deduplicate while preserving order.
        seen = set()
        uniq = []
        for p in candidates:
            key = p.as_posix()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        if not uniq:
            print(f"[MISS] {target}: no candidate")
            continue
        src = uniq[0]
        dst = WEIGHTS_DIR / target
        if args.dry_run:
            print(f"[PLAN] {src.as_posix()} -> {dst.as_posix()}")
        else:
            shutil.copy2(src, dst)
            print(f"[OK] {src.as_posix()} -> {dst.as_posix()}")


if __name__ == "__main__":
    main()
