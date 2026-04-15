from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.paths import WEIGHTS_DIR


def check_dependency(name: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, f"{name}: installed"
    except Exception:
        return False, f"{name}: missing"


def check_weights() -> Tuple[bool, str]:
    expected = [
        WEIGHTS_DIR / "best_teacher.pt",
        WEIGHTS_DIR / "best_student_plain.pt",
        WEIGHTS_DIR / "best_student_ppla.pt",
        WEIGHTS_DIR / "best_student_distill.pt",
    ]
    existing = [p for p in expected if p.exists()]
    if len(existing) >= 2:
        return True, f"weights: {len(existing)}/4 available (>=2 required for switching)"
    return False, f"weights: {len(existing)}/4 available (<2, switch test cannot pass)"


def main() -> None:
    checks: List[Tuple[bool, str]] = [
        check_dependency("flask"),
        check_dependency("ultralytics"),
        check_weights(),
    ]
    ok = all(x[0] for x in checks)

    print("Backend preflight")
    print("")
    for passed, msg in checks:
        mark = "[OK]" if passed else "[FAIL]"
        print(f"{mark} {msg}")
    print("")
    if ok:
        print("Result: PASS")
    else:
        print("Result: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
