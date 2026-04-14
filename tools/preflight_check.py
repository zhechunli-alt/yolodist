from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def check_module(name: str) -> tuple[bool, str]:
    found = importlib.util.find_spec(name) is not None
    return found, f"python module `{name}` {'OK' if found else 'MISSING'}"


def check_path(path: Path, label: str) -> tuple[bool, str]:
    exists = path.exists()
    return exists, f"{label} `{path}` {'OK' if exists else 'MISSING'}"


def main() -> None:
    checks: list[tuple[bool, str]] = []
    for module in ("ultralytics", "torch", "yaml", "PIL"):
        checks.append(check_module(module))

    checks.append(check_path(ROOT / "weights" / "yolo11n.pt", "weights"))
    checks.append(check_path(ROOT / "weights" / "teacher.pt", "weights"))
    checks.append(check_path(ROOT / "datasets" / "mvtec", "dataset dir"))
    checks.append(check_path(ROOT / "configs" / "models" / "yolo11_student.yaml", "model config"))
    checks.append(check_path(ROOT / "configs" / "train" / "modified_model.toml", "train config"))
    checks.append(check_path(ROOT / "configs" / "train" / "distillation.toml", "distill config"))

    all_ok = True
    for ok, msg in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {msg}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nPreflight failed. Fix missing items before training.")
        sys.exit(1)
    print("\nPreflight passed. You can start server-side training.")


if __name__ == "__main__":
    main()

