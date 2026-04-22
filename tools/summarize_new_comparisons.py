#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/root/workspace/yolodist")
OUT_CSV = ROOT / "runs/paper_tables/deeppcb_additional_comparisons.csv"
OUT_MD = ROOT / "runs/paper_tables/deeppcb_additional_comparisons.md"
OUT_FIG = ROOT / "runs/paper_figures/generated/deeppcb_additional_comparisons_map5095.png"

ROWS = [
    ("student_epfa", 1.286, "runs/deeppcb_epfa/student_epfa_eval-*/metrics_summary.json"),
    ("ssdlite320_mobilenet_v3_large", 3.79238, "runs/deeppcb_compare/ssdlite320_mobilenet_v3_large_eval*/metrics_summary.json"),
    ("fasterrcnn_mnv3_320_fpn", 18.955854, "runs/deeppcb_compare/fasterrcnn_mnv3_320_fpn_eval*/metrics_summary.json"),
    ("fasterrcnn_mnv3_fpn", 18.955854, "runs/deeppcb_compare/fasterrcnn_mnv3_fpn_eval*/metrics_summary.json"),
    ("fasterrcnn_r50_fpn", 41.377906, "runs/deeppcb_compare/fasterrcnn_r50_fpn_eval*/metrics_summary.json"),
    ("fasterrcnn_r50_fpn_v2", 43.281778, "runs/deeppcb_compare/fasterrcnn_r50_fpn_v2_eval*/metrics_summary.json"),
]


def load(pattern: str):
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        return None
    with matches[-1].open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    result_rows = []
    for model, params_m, pattern in ROWS:
        data = load(pattern)
        if not data:
            continue
        speed = data.get("speed", {})
        inf_ms = float(speed.get("inference", 0.0)) if speed else 0.0
        result_rows.append(
            {
                "model": model,
                "params_m": params_m,
                "precision": float(data.get("metrics/precision(B)", 0.0)),
                "recall": float(data.get("metrics/recall(B)", 0.0)),
                "map50": float(data.get("metrics/mAP50(B)", 0.0)),
                "map5095": float(data.get("metrics/mAP50-95(B)", 0.0)),
                "inference_ms": inf_ms,
                "fps_infer": (1000.0 / inf_ms) if inf_ms > 0 else 0.0,
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "params_m", "precision", "recall", "map50", "map5095", "inference_ms", "fps_infer"],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    lines = [
        "# DeepPCB 新增对比模型结果",
        "",
        "| 模型 | Params(M) | Precision | Recall | mAP50 | mAP50-95 | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['model']} | {row['params_m']:.3f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['map50']:.4f} | {row['map5095']:.4f} | {row['fps_infer']:.2f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if result_rows:
        plt.figure(figsize=(9, 5))
        palette = ["#3b7a57", "#b85c38", "#7a3e9d", "#1f4e79", "#8c5a2b", "#5a6f8c"]
        plt.bar([row["model"] for row in result_rows], [row["map5095"] for row in result_rows], color=palette[: len(result_rows)])
        plt.ylabel("mAP50-95")
        plt.title("DeepPCB Additional Non-YOLO Comparisons")
        plt.xticks(rotation=15)
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT_FIG, dpi=220)
        plt.close()


if __name__ == "__main__":
    main()
