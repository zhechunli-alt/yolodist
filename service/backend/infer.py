from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os

import numpy as np
from PIL import Image
import torch
import yaml

from .model_registry import ModelRegistry
from src.yolodist.paths import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.yolodist.compare.runner import build_torchvision_model


class InferEngine:
    """Lazy-load YOLO models and run single-image inference."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._model_cache: Dict[str, Any] = {}
        self._warm_cache: set[str] = set()
        self._torchvision_checkpoint_cache: Dict[str, dict[str, Any]] = {}

    def _dataset_data_yaml(self, dataset: str) -> Path:
        mapping = {
            "DeepPCB": ROOT / "datasets/processed/deeppcb_detection/data.yaml",
            "PKU-Market-PCB": ROOT / "datasets/processed/pku_market_pcb_detection/data.yaml",
            "DsPCBSD+": ROOT / "datasets/processed/dspcbsd_plus_detection/data.yaml",
        }
        if dataset not in mapping:
            raise KeyError(f"Unsupported dataset for names: {dataset}")
        return mapping[dataset]

    def _class_names_for_dataset(self, dataset: str) -> dict[int, str]:
        spec = yaml.safe_load(self._dataset_data_yaml(dataset).read_text(encoding="utf-8"))
        names = spec.get("names", {}) or {}
        return {int(k): str(v) for k, v in names.items()}

    def _demo_sample_for_dataset(self, dataset: str) -> Path:
        sample_map = {
            "DeepPCB": ROOT / "assets/demo_samples/deeppcb/single/00041200_test.jpg",
            "PKU-Market-PCB": ROOT / "assets/demo_samples/pku/single/01_open_circuit_03.jpg",
            "DsPCBSD+": ROOT / "assets/demo_samples/dspcbsd_plus/single/0004990.jpg",
        }
        path = sample_map.get(dataset)
        if path is None or not path.exists():
            raise FileNotFoundError(f"No demo sample available for dataset: {dataset}")
        return path

    def _load_model(self, model_name: str, model_path: Path) -> Any:
        cached = self._model_cache.get(model_name)
        if cached is not None:
            return cached
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "False")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        import yolodist  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
        from yolodist.models.registry import (  # pylint: disable=import-outside-toplevel
            register_ultralytics_modules,
        )
        try:
            from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
        except Exception as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc
        register_ultralytics_modules()
        model = YOLO(model_path.as_posix())
        self._model_cache[model_name] = model
        return model

    def _load_torchvision_model(self, model_name: str, model_path: Path) -> tuple[Any, dict[int, str], int]:
        cached = self._model_cache.get(model_name)
        if cached is not None:
            spec = self.registry.get_model_spec(model_name)
            return cached, self._class_names_for_dataset(spec.dataset), self._torchvision_checkpoint_cache[model_name]["imgsz"]

        spec = self.registry.get_model_spec(model_name)
        checkpoint = torch.load(model_path, map_location="cpu")
        ckpt_model_name = str(checkpoint["model_name"])
        num_classes_without_background = int(checkpoint["num_classes"]) - 1
        imgsz = int(checkpoint.get("imgsz", 640))
        model = build_torchvision_model(ckpt_model_name, num_classes_without_background)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self._model_cache[model_name] = model
        self._torchvision_checkpoint_cache[model_name] = {
            "imgsz": imgsz,
            "device": device,
        }
        return model, self._class_names_for_dataset(spec.dataset), imgsz

    def _infer_torchvision(
        self,
        *,
        image_path: Path,
        model_name: str,
        model_path: Path,
        conf: Optional[float],
    ) -> Dict[str, Any]:
        model, class_names, imgsz = self._load_torchvision_model(model_name, model_path)
        ckpt_meta = self._torchvision_checkpoint_cache[model_name]
        device: torch.device = ckpt_meta["device"]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        resized = image.resize((imgsz, imgsz))
        tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(device)
        threshold = 0.25 if conf is None else float(conf)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model([tensor])
        latency_ms = (time.perf_counter() - start) * 1000.0
        output = outputs[0]
        detections: List[Dict[str, Any]] = []
        boxes = output["boxes"].detach().cpu()
        scores = output["scores"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scale_x = orig_w / imgsz
        scale_y = orig_h / imgsz
        for box, score, label in zip(boxes, scores, labels):
            conf_val = float(score.item())
            if conf_val < threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            cls_idx = int(label.item()) - 1
            detections.append(
                {
                    "cls": cls_idx,
                    "class_name": str(class_names.get(cls_idx, cls_idx)),
                    "conf": round(conf_val, 4),
                    "xyxy": [
                        round(float(x1 * scale_x), 2),
                        round(float(y1 * scale_y), 2),
                        round(float(x2 * scale_x), 2),
                        round(float(y2 * scale_y), 2),
                    ],
                }
            )
        return {
            "model_name": model_name,
            "latency_ms": round(latency_ms, 3),
            "detections": detections,
            "image_size": {"width": int(orig_w), "height": int(orig_h)},
        }

    def infer(
        self,
        image_path: str,
        model_name: Optional[str] = None,
        conf: Optional[float] = None,
    ) -> Dict[str, Any]:
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"Image does not exist: {image.as_posix()}")

        target_model = model_name or self.registry.current_model()
        if target_model is None:
            raise RuntimeError(
                "No active model. Put standard weights into weights/ and switch one model first."
            )

        model_path = self.registry.get_model_path(target_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights do not exist for '{target_model}': {model_path.as_posix()}"
            )

        spec = self.registry.get_model_spec(target_model)
        if spec.group == "comparison":
            return self._infer_torchvision(
                image_path=image,
                model_name=target_model,
                model_path=model_path,
                conf=conf,
            )

        model = self._load_model(target_model, model_path)
        class_names = getattr(model, "names", {}) or {}

        start = time.perf_counter()
        results = model.predict(source=image.as_posix(), conf=conf, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if not results:
            return {
                "model_name": target_model,
                "latency_ms": round(latency_ms, 3),
                "detections": [],
                "image_size": None,
            }

        result = results[0]
        detections: List[Dict[str, Any]] = []
        for box in result.boxes:
            cls_val = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "cls": cls_val,
                    "class_name": str(class_names.get(cls_val, cls_val)),
                    "conf": round(conf_val, 4),
                    "xyxy": xyxy,
                }
            )

        h, w = result.orig_shape
        return {
            "model_name": target_model,
            "latency_ms": round(latency_ms, 3),
            "detections": detections,
            "image_size": {"width": int(w), "height": int(h)},
        }

    def warm_for_compare(
        self,
        *,
        image_path: str,
        model_names: List[str],
        conf: Optional[float] = None,
    ) -> None:
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"Image does not exist: {image.as_posix()}")
        for model_name in model_names:
            if model_name in self._warm_cache:
                continue
            model_path = self.registry.get_model_path(model_name)
            spec = self.registry.get_model_spec(model_name)
            if spec.group == "comparison":
                self._infer_torchvision(
                    image_path=image,
                    model_name=model_name,
                    model_path=model_path,
                    conf=conf,
                )
            else:
                model = self._load_model(model_name, model_path)
                model.predict(source=image.as_posix(), conf=conf, verbose=False)
            self._warm_cache.add(model_name)

    def prewarm_model(self, model_name: str, conf: Optional[float] = 0.25) -> None:
        if model_name in self._warm_cache:
            return
        spec = self.registry.get_model_spec(model_name)
        sample = self._demo_sample_for_dataset(spec.dataset)
        self.warm_for_compare(
            image_path=sample.as_posix(),
            model_names=[model_name],
            conf=conf,
        )
