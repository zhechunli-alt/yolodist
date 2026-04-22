from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.yolodist.paths import ROOT, WEIGHTS_DIR


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    exists: bool


class ModelRegistry:
    """Central source of truth for backend model discovery and switching."""

    def __init__(self, weights_dir: Path | None = None) -> None:
        self.weights_dir = weights_dir or WEIGHTS_DIR
        self._standard_model_paths: Dict[str, Path] = {
            "student_epfa": self.weights_dir / "best_deeppcb_student_epfa.pt",
            "baseline": self.weights_dir / "best_deeppcb_baseline.pt",
            "student_plain": self.weights_dir / "best_deeppcb_student_plain.pt",
            "distill_epfa": self.weights_dir / "best_deeppcb_distill_epfa.pt",
            "teacher_deeppcb": self.weights_dir / "teacher_deeppcb.pt",
            "teacher_pku_market_pcb": self.weights_dir / "teacher_pku_market_pcb.pt",
            "teacher_dspcbsd_plus": self.weights_dir / "teacher_dspcbsd_plus.pt",
            "teacher": self.weights_dir / "best_teacher.pt",
            "student_ppla": self.weights_dir / "best_student_ppla.pt",
            "student_distill": self.weights_dir / "best_student_distill.pt",
        }
        self._model_paths: Dict[str, Path] = {}
        self._rebuild_model_paths()
        self._current_model = self._pick_default_model()

    def _rebuild_model_paths(self) -> None:
        merged: Dict[str, Path] = dict(self._standard_model_paths)
        if self.weights_dir.exists():
            for path in sorted(self.weights_dir.glob("*.pt")):
                alias = path.stem
                if alias not in merged:
                    merged[alias] = path
        self._model_paths = merged

    def _pick_default_model(self) -> Optional[str]:
        preferred_order = [
            "student_epfa",
            "distill_epfa",
            "student_plain",
            "baseline",
            "teacher_deeppcb",
        ]
        for name in preferred_order:
            path = self._model_paths.get(name)
            if path and path.exists():
                return name
        for name, path in self._model_paths.items():
            if path.exists():
                return name
        return None

    def refresh(self) -> None:
        self._rebuild_model_paths()
        if self._current_model and not self._model_paths[self._current_model].exists():
            self._current_model = self._pick_default_model()
        if self._current_model is None:
            self._current_model = self._pick_default_model()

    def list_models(self) -> List[ModelSpec]:
        self.refresh()
        return [
            ModelSpec(name=name, path=path, exists=path.exists())
            for name, path in self._model_paths.items()
        ]

    def current_model(self) -> Optional[str]:
        self.refresh()
        return self._current_model

    def current_model_path(self) -> Optional[Path]:
        name = self.current_model()
        if name is None:
            return None
        return self._model_paths[name]

    def get_model_path(self, model_name: str) -> Path:
        if model_name not in self._model_paths:
            raise KeyError(f"Unknown model name: {model_name}")
        return self._model_paths[model_name]

    def switch(self, model_name: str) -> Path:
        if model_name not in self._model_paths:
            raise KeyError(f"Unknown model name: {model_name}")
        path = self._model_paths[model_name]
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights do not exist for '{model_name}': {path.as_posix()}"
            )
        self._current_model = model_name
        return path

    def as_json(self) -> Dict[str, object]:
        models = self.list_models()
        return {
            "root_dir": ROOT.as_posix(),
            "weights_dir": self.weights_dir.as_posix(),
            "current_model": self.current_model(),
            "models": [
                {
                    "name": spec.name,
                    "weights_path": spec.path.as_posix(),
                    "available": spec.exists,
                }
                for spec in models
            ],
        }
