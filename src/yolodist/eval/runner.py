from __future__ import annotations

from pathlib import Path
import json

from yolodist.config import load_toml
from yolodist.models.registry import register_ultralytics_modules
from yolodist.paths import ROOT
from yolodist.reporting.manifest import write_run_manifest


def run_eval(config_path: Path) -> Path:
    config = load_toml(config_path)["eval"]
    model_path = ROOT / config["model"]
    data_path = ROOT / config["data"]
    run_dir = ROOT / config["project"] / config["name"]
    ensure_exists(model_path, "evaluation model")
    ensure_exists(data_path, "dataset yaml")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    register_ultralytics_modules()

    write_run_manifest(
        run_dir=run_dir,
        payload={
            "stage": "eval",
            "config_path": str(config_path),
            "eval": config,
        },
    )

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=str(config.get("split", "test")),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=str(config["device"]),
        project=str(ROOT / config["project"]),
        name=config["name"],
    )

    save_dir = Path(metrics.save_dir)
    summary_path = save_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(normalize_metrics(metrics), indent=2), encoding="utf-8")
    return summary_path


def normalize_metrics(metrics) -> dict[str, object]:
    summary: dict[str, object] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                summary[str(key)] = value
            else:
                summary[str(key)] = str(value)

    for attr in ("fitness", "speed"):
        value = getattr(metrics, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            summary[attr] = value
        elif isinstance(value, (int, float, str, bool)):
            summary[attr] = value
        else:
            summary[attr] = str(value)
    return summary


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
