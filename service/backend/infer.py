from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os

from .model_registry import ModelRegistry
from src.yolodist.paths import ROOT


class InferEngine:
    """Lazy-load YOLO models and run single-image inference."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._model_cache: Dict[str, Any] = {}

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
