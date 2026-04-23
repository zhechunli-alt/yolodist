from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from service.backend.infer import InferEngine
from service.backend.model_registry import ModelRegistry


TEST_DIR = ROOT / "datasets" / "processed" / "deeppcb_detection" / "images" / "test"
DEFAULT_MODELS = [
    "deeppcb_baseline",
    "deeppcb_distill_epfa",
    "cmp_frcnn_r50",
]
MODEL_ALIASES = {
    "deeppcb_baseline": {
        "registry_key": "deeppcb_teacher",
        "display_name": "DeepPCB / Baseline-YOLO11n",
        "group": "baseline",
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def collect_test_images(limit: int | None = None) -> List[Path]:
    images = sorted([*TEST_DIR.glob("*.jpg"), *TEST_DIR.glob("*.png")])
    if limit is not None:
        return images[:limit]
    return images


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    sorted_vals = sorted(latencies)
    mean_val = statistics.fmean(sorted_vals)
    median_val = statistics.median(sorted_vals)
    std_val = statistics.pstdev(sorted_vals) if len(sorted_vals) > 1 else 0.0
    p95 = percentile(sorted_vals, 0.95)
    p99 = percentile(sorted_vals, 0.99)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    fps = 1000.0 / mean_val if mean_val > 0 else math.nan
    return {
        "mean_ms": mean_val,
        "median_ms": median_val,
        "std_ms": std_val,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": min_val,
        "max_ms": max_val,
        "fps": fps,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_key",
        "display_name",
        "group",
        "sample_count",
        "warmup_count",
        "mean_ms",
        "median_ms",
        "std_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "fps",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    sample_count: int,
    warmup_count: int,
) -> None:
    lines = [
        "# DeepPCB 推理耗时统计",
        "",
        f"- 数据集：`DeepPCB test`",
        f"- 样本规模：`{sample_count}` 张",
        f"- 预热张数：`{warmup_count}` 张/模型",
        "- 统计口径：单张推理耗时（`InferEngine.infer()` 返回的 `latency_ms`），不含上传与前端传输时间",
        "",
        "| 模型 | 均值(ms) | 中位数(ms) | P95(ms) | P99(ms) | 最小(ms) | 最大(ms) | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {display_name} | {mean_ms:.3f} | {median_ms:.3f} | {p95_ms:.3f} | {p99_ms:.3f} | {min_ms:.3f} | {max_ms:.3f} | {fps:.2f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DeepPCB inference latency")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on test images")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup images per model")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "paper_tables"),
        help="Directory for CSV/MD/JSON outputs",
    )
    args = parser.parse_args()

    images = collect_test_images(args.limit)
    if not images:
        raise SystemExit("No DeepPCB test images found")

    registry = ModelRegistry()
    engine = InferEngine(registry=registry)

    rows: List[Dict[str, Any]] = []
    raw: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "dataset": "DeepPCB",
        "sample_count": len(images),
        "warmup_count": args.warmup,
        "images": [p.as_posix() for p in images],
        "models": {},
    }

    for model_key in args.models:
        alias = MODEL_ALIASES.get(model_key)
        registry_key = alias["registry_key"] if alias else model_key
        spec = registry.get_model_spec(registry_key)
        warmup_images = images[: min(args.warmup, len(images))]
        for img in warmup_images:
            engine.infer(image_path=img.as_posix(), model_name=registry_key, conf=0.25)

        latencies: List[float] = []
        for img in images:
            out = engine.infer(image_path=img.as_posix(), model_name=registry_key, conf=0.25)
            latencies.append(float(out["latency_ms"]))

        summary = summarize_latencies(latencies)
        row = {
            "model_key": model_key,
            "display_name": alias["display_name"] if alias else spec.display_name,
            "group": alias["group"] if alias else spec.group,
            "sample_count": len(images),
            "warmup_count": len(warmup_images),
            **summary,
        }
        rows.append(row)
        raw["models"][model_key] = {
            "registry_key": registry_key,
            "display_name": alias["display_name"] if alias else spec.display_name,
            "group": alias["group"] if alias else spec.group,
            "latencies_ms": latencies,
            "summary": summary,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "deeppcb_inference_latency_benchmark.csv"
    md_path = output_dir / "deeppcb_inference_latency_benchmark.md"
    json_path = output_dir / "deeppcb_inference_latency_benchmark.json"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows, sample_count=len(images), warmup_count=len(warmup_images))
    json_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
