from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.yolodist.paths import ROOT, WEIGHTS_DIR


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    group: str
    dataset: str
    role: str
    path: Path
    exists: bool
    runtime_supported: bool


class ModelRegistry:
    """Central source of truth for backend model discovery and switching."""

    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir or WEIGHTS_DIR
        self._catalog: Dict[str, Dict[str, object]] = {
            "deeppcb_student_epfa": {
                "display_name": "DeepPCB / Student-EPFA",
                "group": "mainline",
                "dataset": "DeepPCB",
                "role": "demo",
                "path": self.weights_dir / "best_deeppcb_student_epfa.pt",
                "runtime_supported": True,
            },
            "deeppcb_baseline": {
                "display_name": "DeepPCB / Baseline-YOLO11n",
                "group": "mainline",
                "dataset": "DeepPCB",
                "role": "baseline",
                "path": self.weights_dir / "best_deeppcb_baseline.pt",
                "runtime_supported": True,
            },
            "deeppcb_student_plain": {
                "display_name": "DeepPCB / Student-Plain",
                "group": "mainline",
                "dataset": "DeepPCB",
                "role": "student",
                "path": self.weights_dir / "best_deeppcb_student_plain.pt",
                "runtime_supported": True,
            },
            "deeppcb_distill_epfa": {
                "display_name": "DeepPCB / Distill-EPFA",
                "group": "mainline",
                "dataset": "DeepPCB",
                "role": "distill",
                "path": self.weights_dir / "best_deeppcb_distill_epfa.pt",
                "runtime_supported": True,
            },
            "deeppcb_teacher": {
                "display_name": "DeepPCB / Teacher",
                "group": "teacher",
                "dataset": "DeepPCB",
                "role": "teacher",
                "path": self.weights_dir / "teacher_deeppcb.pt",
                "runtime_supported": True,
            },
            "pku_teacher": {
                "display_name": "PKU / Teacher",
                "group": "teacher",
                "dataset": "PKU-Market-PCB",
                "role": "teacher",
                "path": self.weights_dir / "teacher_pku_market_pcb.pt",
                "runtime_supported": True,
            },
            "dspcbsd_teacher": {
                "display_name": "DsPCBSD+ / Teacher",
                "group": "teacher",
                "dataset": "DsPCBSD+",
                "role": "teacher",
                "path": self.weights_dir / "teacher_dspcbsd_plus.pt",
                "runtime_supported": True,
            },
            "cmp_ssdlite_mnv3": {
                "display_name": "Compare / SSDLite-MobileNetV3",
                "group": "comparison",
                "dataset": "DeepPCB",
                "role": "comparison",
                "path": ROOT / "runs/deeppcb_compare/ssdlite320_mobilenet_v3_large/weights/best.pt",
                "runtime_supported": False,
            },
            "cmp_frcnn_mnv3_320": {
                "display_name": "Compare / FasterRCNN-MNV3-320",
                "group": "comparison",
                "dataset": "DeepPCB",
                "role": "comparison",
                "path": ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn/weights/best.pt",
                "runtime_supported": False,
            },
            "cmp_frcnn_mnv3_fpn": {
                "display_name": "Compare / FasterRCNN-MNV3-FPN",
                "group": "comparison",
                "dataset": "DeepPCB",
                "role": "comparison",
                "path": ROOT / "runs/deeppcb_compare/fasterrcnn_mnv3_fpn/weights/best.pt",
                "runtime_supported": False,
            },
            "cmp_retinanet_r50": {
                "display_name": "Compare / RetinaNet-R50-FPN",
                "group": "comparison",
                "dataset": "DeepPCB",
                "role": "comparison",
                "path": ROOT / "runs/deeppcb_compare/retinanet_r50_fpn/weights/best.pt",
                "runtime_supported": False,
            },
            "cmp_fcos_r50": {
                "display_name": "Compare / FCOS-R50-FPN",
                "group": "comparison",
                "dataset": "DeepPCB",
                "role": "comparison",
                "path": ROOT / "runs/deeppcb_compare/fcos_r50_fpn/weights/best.pt",
                "runtime_supported": False,
            },
            "cmp_frcnn_r50": {
                "display_name": "Compare / FasterRCNN-R50-FPN",
                "group": "comparison",
                "dataset": "DeepPCB",
                "role": "comparison",
                "path": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn/weights/best.pt",
                "runtime_supported": False,
            },
            "cmp_frcnn_r50_v2": {
                "display_name": "Compare / FasterRCNN-R50-FPN-v2",
                "group": "comparison",
                "dataset": "DeepPCB",
                "role": "comparison",
                "path": ROOT / "runs/deeppcb_compare/fasterrcnn_r50_fpn_v2/weights/best.pt",
                "runtime_supported": False,
            },
        }
        self._specs: Dict[str, ModelSpec] = {}
        self._rebuild_model_paths()
        self._current_model = self._pick_default_model()

    def _rebuild_model_paths(self) -> None:
        self._specs = {}
        for key, item in self._catalog.items():
            path = Path(item["path"])
            self._specs[key] = ModelSpec(
                key=key,
                display_name=str(item["display_name"]),
                group=str(item["group"]),
                dataset=str(item["dataset"]),
                role=str(item["role"]),
                path=path,
                exists=path.exists(),
                runtime_supported=bool(item.get("runtime_supported", True)),
            )

    def _pick_default_model(self) -> Optional[str]:
        preferred_order = [
            "deeppcb_student_epfa",
            "deeppcb_distill_epfa",
            "deeppcb_student_plain",
            "deeppcb_baseline",
            "deeppcb_teacher",
        ]
        for key in preferred_order:
            spec = self._specs.get(key)
            if spec and spec.exists:
                return key
        for key, spec in self._specs.items():
            if spec.exists:
                return key
        return None

    def refresh(self) -> None:
        self._rebuild_model_paths()
        if self._current_model and not self._specs[self._current_model].exists:
            self._current_model = self._pick_default_model()
        if self._current_model is None:
            self._current_model = self._pick_default_model()

    def list_models(self, *, include_missing: bool = False) -> List[ModelSpec]:
        self.refresh()
        models = list(self._specs.values())
        if not include_missing:
            models = [spec for spec in models if spec.exists]
        return models

    def current_model(self) -> Optional[str]:
        self.refresh()
        return self._current_model

    def current_model_spec(self) -> Optional[ModelSpec]:
        name = self.current_model()
        if name is None:
            return None
        return self._specs[name]

    def current_model_path(self) -> Optional[Path]:
        spec = self.current_model_spec()
        return spec.path if spec else None

    def get_model_path(self, model_key: str) -> Path:
        if model_key not in self._specs:
            raise KeyError(f"Unknown model key: {model_key}")
        return self._specs[model_key].path

    def is_runtime_supported(self, model_key: str) -> bool:
        if model_key not in self._specs:
            raise KeyError(f"Unknown model key: {model_key}")
        return self._specs[model_key].runtime_supported

    def switch(self, model_key: str) -> Path:
        if model_key not in self._specs:
            raise KeyError(f"Unknown model key: {model_key}")
        spec = self._specs[model_key]
        if not spec.exists:
            raise FileNotFoundError(
                f"Model weights do not exist for '{model_key}': {spec.path.as_posix()}"
            )
        if not spec.runtime_supported:
            raise RuntimeError(
                f"Model '{model_key}' is archived for paper comparison only and is not available for online inference."
            )
        self._current_model = model_key
        return spec.path

    def as_json(self) -> Dict[str, object]:
        models = self.list_models()
        current = self.current_model_spec()
        grouped: Dict[str, List[Dict[str, object]]] = {"mainline": [], "comparison": [], "teacher": []}
        payload_models = []
        for spec in models:
            item = {
                "key": spec.key,
                "display_name": spec.display_name,
                "group": spec.group,
                "dataset": spec.dataset,
                "role": spec.role,
                "weights_path": spec.path.as_posix(),
                "available": spec.exists,
                "runtime_supported": spec.runtime_supported,
            }
            grouped.setdefault(spec.group, []).append(item)
            payload_models.append(item)
        return {
            "root_dir": ROOT.as_posix(),
            "weights_dir": self.weights_dir.as_posix(),
            "current_model": current.key if current else None,
            "current_display_name": current.display_name if current else None,
            "models": payload_models,
            "groups": grouped,
        }
