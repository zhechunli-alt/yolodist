from __future__ import annotations

from pathlib import Path

from yolodist.config import load_toml
from yolodist.models.registry import initialize_custom_model_context, register_ultralytics_modules
from yolodist.paths import ROOT
from yolodist.reporting.manifest import write_run_manifest


def run_train(config_path: Path) -> None:
    config = load_toml(config_path)["train"]
    model_path = ROOT / config["model"]
    data_path = ROOT / config["data"]
    run_dir = ROOT / config["project"] / config["name"]
    ensure_exists(model_path, "model weights")
    ensure_exists(data_path, "dataset yaml")
    if "pretrained" in config:
        ensure_exists(ROOT / config["pretrained"], "pretrained checkpoint")

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
            "stage": "train",
            "config_path": str(config_path),
            "train": config,
        },
    )

    model = YOLO(str(model_path))
    initialize_custom_model_context(model.model)
    train_kwargs = {
        "data": str(data_path),
        "project": str(ROOT / config["project"]),
        "name": config["name"],
        "epochs": int(config["epochs"]),
        "imgsz": int(config["imgsz"]),
        "batch": int(config["batch"]),
        "device": str(config["device"]),
        "workers": int(config["workers"]),
        "seed": int(config["seed"]),
        "exist_ok": bool(config["exist_ok"]),
    }
    if "pretrained" in config:
        train_kwargs["pretrained"] = str(ROOT / config["pretrained"])
    if "optimizer" in config:
        train_kwargs["optimizer"] = str(config["optimizer"])
    if "lr0" in config:
        train_kwargs["lr0"] = float(config["lr0"])
    if "momentum" in config:
        train_kwargs["momentum"] = float(config["momentum"])
    if "resume" in config:
        train_kwargs["resume"] = bool(config["resume"])
    model.train(
        **train_kwargs
    )


def run_predict(model_path: Path, image_path: Path, project: Path, name: str) -> None:
    ensure_exists(model_path, "model weights")
    ensure_exists(image_path, "input image")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    register_ultralytics_modules()

    model = YOLO(str(model_path))
    initialize_custom_model_context(model.model)
    model(str(image_path), save=True, project=str(project), name=name)


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
