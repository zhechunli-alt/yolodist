#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/root/workspace/yolodist")
RUNS = ROOT / "runs"
TABLE_DIR = RUNS / "paper_tables"
FIG_DIR = RUNS / "paper_figures"
GEN_DIR = FIG_DIR / "generated"


MAINLINE_CONFIG = {
    "DeepPCB": {
        "baseline": {
            "params_m": 2.624,
            "metrics": RUNS / "deeppcb_baseline/baseline_plain_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "deeppcb_baseline_pred.jpg",
        },
        "student_plain": {
            "params_m": 1.286,
            "metrics": RUNS / "deeppcb_modified/student_plain_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "deeppcb_student_plain_pred.jpg",
        },
        "student_epfa": {
            "params_m": 1.286,
            "metrics": RUNS / "deeppcb_epfa/student_epfa_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "deeppcb_student_epfa_pred.jpg",
        },
        "distill_epfa_fair": {
            "params_m": 1.286,
            "metrics": RUNS / "deeppcb_distill_epfa/distill_epfa_fair_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "deeppcb_distill_epfa_fair_pred.jpg",
        },
        "distill_epfa_tuned": {
            "params_m": 1.286,
            "metrics": RUNS / "deeppcb_distill_epfa_tuned/distill_epfa_tuned_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "deeppcb_distill_epfa_tuned_pred.jpg",
        },
    },
    "PKU-Market-PCB": {
        "baseline": {
            "params_m": 2.624,
            "metrics": RUNS / "pku_market_pcb_baseline/baseline_plain_eval-2/metrics_summary.json",
            "pred": RUNS / "pku_market_pcb_baseline/baseline_plain_eval-2/val_batch0_pred.jpg",
        },
        "student_plain": {
            "params_m": 1.286,
            "metrics": RUNS / "pku_market_pcb_modified/student_plain_eval-3/metrics_summary.json",
            "pred": FIG_DIR / "pku_student_plain_pred.jpg",
        },
        "student_epfa": {
            "params_m": 1.286,
            "metrics": RUNS / "pku_market_pcb_epfa/student_epfa_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "pku_student_epfa_pred.jpg",
        },
        "distill_epfa_fair": {
            "params_m": 1.286,
            "metrics": RUNS / "pku_market_pcb_distill_epfa/distill_epfa_fair_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "pku_distill_epfa_fair_pred.jpg",
        },
        "distill_epfa_tuned": {
            "params_m": 1.286,
            "metrics": RUNS / "pku_market_pcb_distill_epfa_tuned/distill_epfa_tuned_eval-3/metrics_summary.json",
            "pred": FIG_DIR / "pku_distill_epfa_tuned_pred.jpg",
        },
    },
    "DsPCBSD+": {
        "baseline": {
            "params_m": 2.592,
            "metrics": RUNS / "dspcbsd_plus_baseline/baseline_plain_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "dspcbsd_baseline_pred.jpg",
        },
        "student_plain": {
            "params_m": 1.286,
            "metrics": RUNS / "dspcbsd_plus_modified/student_plain_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "dspcbsd_student_plain_pred.jpg",
        },
        "student_epfa": {
            "params_m": 1.286,
            "metrics": RUNS / "dspcbsd_plus_epfa/student_epfa_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "dspcbsd_student_epfa_pred.jpg",
        },
        "distill_epfa_fair": {
            "params_m": 1.286,
            "metrics": RUNS / "dspcbsd_plus_distill_epfa/distill_epfa_fair_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "dspcbsd_distill_epfa_fair_pred.jpg",
        },
        "distill_epfa_tuned": {
            "params_m": 1.286,
            "metrics": RUNS / "dspcbsd_plus_distill_epfa_tuned/distill_epfa_tuned_eval-2/metrics_summary.json",
            "pred": FIG_DIR / "dspcbsd_distill_epfa_tuned_pred.jpg",
        },
    },
}

MODEL_ORDER_MAIN = ["baseline", "student_plain", "student_epfa", "distill_epfa_fair"]
MODEL_ORDER_DISTILL = ["student_epfa", "distill_epfa_fair", "distill_epfa_tuned"]
COLORS = {
    "baseline": "#1f4e79",
    "student_plain": "#b85c38",
    "student_epfa": "#3b7a57",
    "distill_epfa_fair": "#7a3e9d",
    "distill_epfa_tuned": "#c98b00",
}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metrics_entry(path: Path, params_m: float) -> Dict[str, float]:
    raw = load_json(path)
    inference_ms = float(raw["speed"]["inference"])
    total_ms = (
        float(raw["speed"]["preprocess"])
        + float(raw["speed"]["inference"])
        + float(raw["speed"]["postprocess"])
    )
    return {
        "precision": float(raw["metrics/precision(B)"]),
        "recall": float(raw["metrics/recall(B)"]),
        "map50": float(raw["metrics/mAP50(B)"]),
        "map5095": float(raw["metrics/mAP50-95(B)"]),
        "inference_ms": inference_ms,
        "fps_infer": 1000.0 / inference_ms if inference_ms > 0 else math.nan,
        "fps_end2end": 1000.0 / total_ms if total_ms > 0 else math.nan,
        "params_m": params_m,
    }


def collect_rows() -> List[Dict]:
    rows = []
    for dataset, dataset_cfg in MAINLINE_CONFIG.items():
        for model, cfg in dataset_cfg.items():
            row = {"dataset": dataset, "model": model}
            row.update(metrics_entry(cfg["metrics"], cfg["params_m"]))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: float) -> str:
    return f"{v:.4f}"


def write_tables(rows: List[Dict]) -> None:
    main_rows = [
        row
        for row in rows
        if row["model"] in {"baseline", "student_plain", "student_epfa", "distill_epfa_fair"}
    ]
    write_csv(
        TABLE_DIR / "pcb_main_results.csv",
        main_rows,
        [
            "dataset",
            "model",
            "params_m",
            "precision",
            "recall",
            "map50",
            "map5095",
            "inference_ms",
            "fps_infer",
            "fps_end2end",
        ],
    )

    by_key = {(r["dataset"], r["model"]): r for r in rows}
    ablation_rows = []
    for dataset in MAINLINE_CONFIG:
        sp = by_key[(dataset, "student_plain")]
        se = by_key[(dataset, "student_epfa")]
        fair = by_key[(dataset, "distill_epfa_fair")]
        tuned = by_key[(dataset, "distill_epfa_tuned")]
        ablation_rows.append(
            {
                "dataset": dataset,
                "student_plain_map5095": fmt(sp["map5095"]),
                "student_epfa_map5095": fmt(se["map5095"]),
                "epfa_gain_vs_plain": fmt(se["map5095"] - sp["map5095"]),
                "fair_map5095": fmt(fair["map5095"]),
                "fair_gain_vs_epfa": fmt(fair["map5095"] - se["map5095"]),
                "tuned_map5095": fmt(tuned["map5095"]),
                "tuned_gain_vs_epfa": fmt(tuned["map5095"] - se["map5095"]),
                "student_plain_recall": fmt(sp["recall"]),
                "student_epfa_recall": fmt(se["recall"]),
                "fair_recall": fmt(fair["recall"]),
                "tuned_recall": fmt(tuned["recall"]),
            }
        )
    write_csv(
        TABLE_DIR / "pcb_ablation.csv",
        ablation_rows,
        [
            "dataset",
            "student_plain_map5095",
            "student_epfa_map5095",
            "epfa_gain_vs_plain",
            "fair_map5095",
            "fair_gain_vs_epfa",
            "tuned_map5095",
            "tuned_gain_vs_epfa",
            "student_plain_recall",
            "student_epfa_recall",
            "fair_recall",
            "tuned_recall",
        ],
    )


def _font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_panel(title: str, image_paths: List[Path], labels: List[str], out_path: Path) -> None:
    images = [Image.open(p).convert("RGB") for p in image_paths if p.exists()]
    valid_labels = [l for p, l in zip(image_paths, labels) if p.exists()]
    if not images:
        return

    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    cols = 2 if len(images) > 2 else len(images)
    rows = math.ceil(len(images) / cols)
    pad = 24
    header_h = 64
    label_h = 36
    canvas = Image.new(
        "RGB",
        (cols * max_w + (cols + 1) * pad, header_h + rows * (max_h + label_h) + (rows + 1) * pad),
        color=(248, 246, 241),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    label_font = _font(20)
    draw.text((pad, 16), title, fill=(20, 20, 20), font=title_font)

    for idx, (img, label) in enumerate(zip(images, valid_labels)):
        r = idx // cols
        c = idx % cols
        x = pad + c * (max_w + pad)
        y = header_h + pad + r * (max_h + label_h)
        canvas.paste(img.resize((max_w, max_h)), (x, y))
        draw.rectangle([x, y + max_h, x + max_w, y + max_h + label_h], fill=(236, 232, 224))
        draw.text((x + 10, y + max_h + 7), label, fill=(30, 30, 30), font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_panels() -> None:
    for dataset, cfg in MAINLINE_CONFIG.items():
        slug = dataset.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")
        build_panel(
            f"{dataset} Mainline Qualitative Comparison",
            [cfg[m]["pred"] for m in MODEL_ORDER_MAIN],
            ["baseline", "student_plain", "student_epfa", "distill_epfa_fair"],
            GEN_DIR / f"{slug}_qualitative_mainline.png",
        )
        build_panel(
            f"{dataset} Distillation Strategy Comparison",
            [cfg[m]["pred"] for m in MODEL_ORDER_DISTILL],
            ["student_epfa", "distill_epfa_fair", "distill_epfa_tuned"],
            GEN_DIR / f"{slug}_qualitative_distill.png",
        )


def _dataset_rows(rows: List[Dict], dataset: str, models: List[str]) -> List[Dict]:
    lookup = {(r["dataset"], r["model"]): r for r in rows}
    return [lookup[(dataset, m)] for m in models]


def grouped_bar(
    rows: List[Dict],
    metric: str,
    out_path: Path,
    models: List[str],
    title: str,
    ylabel: str,
) -> None:
    datasets = list(MAINLINE_CONFIG.keys())
    x = range(len(datasets))
    width = 0.18 if len(models) == 4 else 0.22
    plt.figure(figsize=(11, 6))
    for idx, model in enumerate(models):
        values = [next(r for r in rows if r["dataset"] == ds and r["model"] == model)[metric] for ds in datasets]
        offset = (idx - (len(models) - 1) / 2) * width
        plt.bar([p + offset for p in x], values, width=width, color=COLORS[model], label=model)
    plt.xticks(list(x), datasets, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def scatter_tradeoff(rows: List[Dict], out_path: Path, x_metric: str, y_metric: str, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 7))
    for row in rows:
        if row["model"] not in {"baseline", "student_plain", "student_epfa", "distill_epfa_fair"}:
            continue
        marker = {"DeepPCB": "o", "PKU-Market-PCB": "s", "DsPCBSD+": "^"}[row["dataset"]]
        plt.scatter(row[x_metric], row[y_metric], s=140, color=COLORS[row["model"]], marker=marker, alpha=0.85)
        label = f'{row["dataset"]}-{row["model"]}'
        plt.annotate(label, (row[x_metric], row[y_metric]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_charts(rows: List[Dict]) -> None:
    main_rows = [r for r in rows if r["model"] in set(MODEL_ORDER_MAIN)]
    grouped_bar(
        main_rows,
        "map5095",
        GEN_DIR / "mainline_map5095_grouped.png",
        MODEL_ORDER_MAIN,
        "Mainline Comparison Across PCB Datasets",
        "mAP50-95",
    )
    grouped_bar(
        main_rows,
        "recall",
        GEN_DIR / "mainline_recall_grouped.png",
        MODEL_ORDER_MAIN,
        "Recall Comparison Across PCB Datasets",
        "Recall",
    )
    grouped_bar(
        [r for r in rows if r["model"] in set(MODEL_ORDER_DISTILL)],
        "map5095",
        GEN_DIR / "distill_strategy_map5095_grouped.png",
        MODEL_ORDER_DISTILL,
        "Distillation Strategy Comparison",
        "mAP50-95",
    )
    grouped_bar(
        [r for r in rows if r["model"] in set(MODEL_ORDER_DISTILL)],
        "recall",
        GEN_DIR / "distill_strategy_recall_grouped.png",
        MODEL_ORDER_DISTILL,
        "Distillation Strategy Recall Comparison",
        "Recall",
    )
    scatter_tradeoff(
        main_rows,
        GEN_DIR / "params_vs_map5095_scatter.png",
        "params_m",
        "map5095",
        "Parameter Efficiency Trade-off",
        "Params (Millions)",
        "mAP50-95",
    )
    scatter_tradeoff(
        main_rows,
        GEN_DIR / "fps_vs_map5095_scatter.png",
        "fps_infer",
        "map5095",
        "Speed-Accuracy Trade-off",
        "FPS (inference only)",
        "mAP50-95",
    )


def write_figure_guide(rows: List[Dict]) -> None:
    lines = [
        "# 论文图表说明",
        "",
        "这份文档用于说明 `runs/paper_figures/` 与 `runs/paper_figures/generated/` 中各类图片的用途、推荐章节位置和可直接使用的图注草稿。",
        "",
        "## 推荐优先级",
        "",
        "1. 三个数据集的主线定性对比大图",
        "2. 主线总结果柱状图",
        "3. 参数量-精度散点图",
        "4. 蒸馏策略对比图",
        "5. 混淆矩阵、PR 曲线、训练收敛曲线",
        "",
        "## 图文件与建议用途",
        "",
        f"- [DeepPCB 主线定性对比](./generated/deeppcb_qualitative_mainline.png)：适合放在实验结果章节，展示 baseline、student_plain、student_epfa、distill_epfa_fair 的检测框差异。",
        f"- [PKU 主线定性对比](./generated/pku_market_pcb_qualitative_mainline.png)：适合放在跨数据集泛化结果章节。",
        f"- [DsPCBSD+ 主线定性对比](./generated/dspcbsdplus_qualitative_mainline.png)：适合强调复杂类别场景下的检测难点。",
        f"- [DeepPCB 蒸馏策略对比](./generated/deeppcb_qualitative_distill.png)：适合放在蒸馏消融章节，比较 student_epfa、fair、tuned 三者。",
        f"- [PKU 蒸馏策略对比](./generated/pku_market_pcb_qualitative_distill.png)：适合说明蒸馏对召回的影响。",
        f"- [DsPCBSD+ 蒸馏策略对比](./generated/dspcbsdplus_qualitative_distill.png)：适合说明 tuned 策略没有稳定优于 fair。",
        f"- [主线 mAP50-95 柱状图](./generated/mainline_map5095_grouped.png)：适合主结果章节。",
        f"- [主线 Recall 柱状图](./generated/mainline_recall_grouped.png)：适合与 mAP50-95 搭配分析。",
        f"- [蒸馏策略 mAP50-95 柱状图](./generated/distill_strategy_map5095_grouped.png)：适合蒸馏消融章节。",
        f"- [蒸馏策略 Recall 柱状图](./generated/distill_strategy_recall_grouped.png)：适合解释蒸馏是否换来更高召回。",
        f"- [参数量-精度散点图](./generated/params_vs_map5095_scatter.png)：适合强调轻量化价值。",
        f"- [速度-精度散点图](./generated/fps_vs_map5095_scatter.png)：适合部署导向讨论。",
        "",
        "## 单图图注草稿",
        "",
        "### 图注 1：主线定性对比",
        "",
        "图X展示了三个 PCB 数据集上的主线定性检测结果。相比 student_plain，student_epfa 在小缺陷与边缘细节定位上更稳定；distill_epfa_fair 在部分样例上提高了召回，但总体上未稳定超过 student_epfa。",
        "",
        "### 图注 2：主线总结果柱状图",
        "",
        "图X对比了 baseline、student_plain、student_epfa 与 distill_epfa_fair 在三个 PCB 数据集上的 mAP50-95 和 Recall。结果表明，baseline 仍然是最强上界，而 EPFA 在 DeepPCB 与 PKU-Market-PCB 上能稳定提升轻量学生模型。",
        "",
        "### 图注 3：参数量-精度散点图",
        "",
        "图X展示了不同模型在参数量与 mAP50-95 之间的权衡关系。student_epfa 在仅约 1.286M 参数量下取得了具有竞争力的检测性能，说明 EPFA-Lite 对轻量模型具有实际价值。",
        "",
        "### 图注 4：蒸馏策略对比",
        "",
        "图X比较了 student_epfa、distill_epfa_fair 与 distill_epfa_tuned 三种策略。fair 版在当前实验中整体稳定性优于 tuned 版，而 tuned 版未能在三个数据集上形成一致增益。",
        "",
        "## 结果摘要",
        "",
    ]
    for dataset in MAINLINE_CONFIG:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        best = max([r for r in ds_rows if r["model"] in set(MODEL_ORDER_MAIN)], key=lambda x: x["map5095"])
        lines.append(
            f"- `{dataset}`：主线最佳模型为 `{best['model']}`，mAP50-95=`{best['map5095']:.4f}`，Recall=`{best['recall']:.4f}`。"
        )
    (FIG_DIR / "FIGURE_GUIDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    write_tables(rows)
    make_panels()
    make_charts(rows)
    write_figure_guide(rows)
    print(f"Generated paper assets into {GEN_DIR}")


if __name__ == "__main__":
    main()
