from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from yolodist.paths import WEIGHTS_DIR

from service.backend.model_registry import ModelRegistry


def check_dependency(name: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, f"{name}: installed"
    except Exception:
        return False, f"{name}: missing"


def check_weights() -> Tuple[bool, str]:
    registry = ModelRegistry(weights_dir=WEIGHTS_DIR)
    available = [spec for spec in registry.list_models() if spec.exists]
    current = registry.current_model()
    if len(available) >= 1:
        preview = ", ".join(spec.name for spec in available[:6])
        return True, f"weights: {len(available)} available; current={current}; models={preview}"
    return False, "weights: 0 available in weights/"


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
