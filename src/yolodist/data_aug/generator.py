from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Protocol

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SyntheticRequest:
    prompt_id: str
    prompt: str
    category: str
    defect_type: str
    severity: str
    class_id: int = 0


@dataclass(frozen=True)
class GenerateResult:
    image_path: Path
    label_path: Path
    quality_score: float


class SyntheticGenerator(Protocol):
    def generate(
        self,
        request: SyntheticRequest,
        dst_image: Path,
        dst_label: Path,
        rng: random.Random,
    ) -> GenerateResult:
        ...


class MockSyntheticGenerator:
    """Local mock generator that creates synthetic-like images plus YOLO labels."""

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 640,
        min_quality: float = 0.45,
        max_quality: float = 0.98,
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        self.image_width = image_width
        self.image_height = image_height
        self.min_quality = min_quality
        self.max_quality = max_quality

    def generate(
        self,
        request: SyntheticRequest,
        dst_image: Path,
        dst_label: Path,
        rng: random.Random,
    ) -> GenerateResult:
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        image = Image.new(
            "RGB",
            (self.image_width, self.image_height),
            color=(rng.randint(170, 220), rng.randint(170, 220), rng.randint(170, 220)),
        )
        draw = ImageDraw.Draw(image)

        # Draw a pseudo defect region as a rectangle; label uses this rectangle.
        x1 = rng.randint(self.image_width // 8, self.image_width // 2)
        y1 = rng.randint(self.image_height // 8, self.image_height // 2)
        x2 = min(self.image_width - 1, x1 + rng.randint(self.image_width // 12, self.image_width // 3))
        y2 = min(self.image_height - 1, y1 + rng.randint(self.image_height // 12, self.image_height // 3))
        draw.rectangle([x1, y1, x2, y2], outline=(180, 30, 30), width=4)

        image.save(dst_image)

        center_x = ((x1 + x2) / 2.0) / self.image_width
        center_y = ((y1 + y2) / 2.0) / self.image_height
        box_w = (x2 - x1) / self.image_width
        box_h = (y2 - y1) / self.image_height
        dst_label.write_text(
            f"{request.class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n",
            encoding="utf-8",
        )

        severity_bias = {
            "minor": 0.10,
            "moderate": 0.0,
            "severe": -0.08,
        }.get(request.severity.lower(), 0.0)
        quality_score = max(
            0.0,
            min(1.0, rng.uniform(self.min_quality, self.max_quality) + severity_bias),
        )

        return GenerateResult(
            image_path=dst_image,
            label_path=dst_label,
            quality_score=quality_score,
        )


def build_generator(config: dict) -> SyntheticGenerator:
    provider = str(config.get("provider", "mock")).strip().lower()
    if provider != "mock":
        raise ValueError(
            f"Unsupported generator provider: {provider}. "
            "Only `mock` is implemented now; real provider can be added later."
        )

    return MockSyntheticGenerator(
        image_width=int(config.get("image_width", 640)),
        image_height=int(config.get("image_height", 640)),
        min_quality=float(config.get("min_quality", 0.45)),
        max_quality=float(config.get("max_quality", 0.98)),
    )
