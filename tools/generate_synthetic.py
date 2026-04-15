from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.config import load_toml
from yolodist.data_aug import (
    MVTecPromptLibrary,
    ManifestRecord,
    SyntheticRequest,
    build_generator,
    normalize_severities,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic MVTec samples with prompt templates.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "data_aug" / "mvtec_mock_v1.toml",
        help="Path to data augmentation TOML config.",
    )
    return parser.parse_args()


def resolve_output_dir(dataset_cfg: dict, config_path: Path) -> Path:
    output_dir = dataset_cfg.get("output_dir")
    if output_dir:
        return ROOT / str(output_dir)
    version = dataset_cfg.get("version", config_path.stem)
    return ROOT / "datasets" / "synthetic" / str(version)


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)

    dataset_cfg = cfg.get("dataset", {})
    prompt_cfg = cfg.get("prompt", {})
    generator_cfg = cfg.get("generator", {})
    quality_cfg = cfg.get("quality", {})
    defect_cfg = cfg.get("defects", [])

    if str(dataset_cfg.get("name", "")).lower() != "mvtec":
        raise ValueError("This generator currently supports dataset.name = 'mvtec' only.")
    if not defect_cfg:
        raise ValueError("No [[defects]] entries found in config.")

    output_dir = resolve_output_dir(dataset_cfg, args.config)
    overwrite = bool(dataset_cfg.get("overwrite", False))
    seed = int(dataset_cfg.get("seed", 42))

    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory exists: {output_dir}. Set dataset.overwrite = true to recreate.")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    prompt_lib = MVTecPromptLibrary(style_hint=str(prompt_cfg.get("style_hint", "industrial inspection photo")))
    generator = build_generator(
        {
            "provider": generator_cfg.get("provider", "mock"),
            "image_width": generator_cfg.get("image_width", 640),
            "image_height": generator_cfg.get("image_height", 640),
            "min_quality": quality_cfg.get("mock_min", 0.45),
            "max_quality": quality_cfg.get("mock_max", 0.98),
        }
    )

    severities = normalize_severities(prompt_cfg.get("severities"))
    default_samples_per_severity = int(prompt_cfg.get("default_samples_per_severity", 4))
    accept_threshold = float(quality_cfg.get("auto_accept_threshold", 0.65))

    records: list[ManifestRecord] = []
    sample_counter = 0

    for group in defect_cfg:
        category = str(group["category"])
        defect_types = group.get("defect_types", [])
        class_id = int(group.get("class_id", 0))
        samples_per_severity = int(group.get("samples_per_severity", default_samples_per_severity))

        for defect_type in defect_types:
            for severity in severities:
                for _ in range(samples_per_severity):
                    sample_counter += 1
                    prompt = prompt_lib.build(category, str(defect_type), severity, sample_counter)
                    stem = prompt.prompt_id
                    image_path = images_dir / f"{stem}.png"
                    label_path = labels_dir / f"{stem}.txt"

                    result = generator.generate(
                        request=SyntheticRequest(
                            prompt_id=prompt.prompt_id,
                            prompt=prompt.prompt,
                            category=category,
                            defect_type=str(defect_type),
                            severity=severity,
                            class_id=class_id,
                        ),
                        dst_image=image_path,
                        dst_label=label_path,
                        rng=rng,
                    )

                    records.append(
                        ManifestRecord(
                            image_path=str(result.image_path.relative_to(ROOT)),
                            label_path=str(result.label_path.relative_to(ROOT)),
                            source="synthetic",
                            prompt_id=prompt.prompt_id,
                            defect_type=f"{category}::{defect_type}",
                            quality_score=result.quality_score,
                            accepted=result.quality_score >= accept_threshold,
                        )
                    )

    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(manifest_path, records)

    accepted = sum(1 for row in records if row.accepted)
    print(f"Generated {len(records)} synthetic samples.")
    print(f"Accepted by auto threshold ({accept_threshold:.2f}): {accepted}/{len(records)}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
