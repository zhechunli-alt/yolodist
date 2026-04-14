from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)
