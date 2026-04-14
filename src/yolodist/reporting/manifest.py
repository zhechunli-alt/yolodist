from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import socket


def write_run_manifest(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    merged = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        **payload,
    }
    out_path = run_dir / "run_manifest.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

