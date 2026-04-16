from __future__ import annotations

import argparse
from pathlib import Path
import sys
import csv
import json


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yolodist.config import load_toml
from yolodist.models.registry import initialize_custom_model_context, register_ultralytics_modules
from yolodist.reporting.summary import summarize_run, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an Ultralytics run directory.")
    parser.add_argument("run_dir", type=Path, nargs="?", help="Path to a training or evaluation run directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output markdown path. Defaults to <run_dir>/summary.md",
    )
    parser.add_argument("--pcb-table-root", type=Path, default=None, help="Root `runs/` directory to scan for PCB result tables.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for generated PCB result tables.")
    args = parser.parse_args()

    if args.pcb_table_root is not None:
        output_dir = (args.output_dir or (args.pcb_table_root / "paper_tables")).resolve()
        generate_pcb_tables(args.pcb_table_root.resolve(), output_dir)
        print(f"PCB tables written to: {output_dir}")
        return

    if args.run_dir is None:
        raise SystemExit("Either run_dir or --pcb-table-root is required.")

    run_dir = args.run_dir.resolve()
    summary = summarize_run(run_dir)
    markdown = to_markdown(summary)
    output = args.output or (run_dir / "summary.md")
    output.write_text(markdown, encoding="utf-8")
    print(f"Summary written to: {output}")


def generate_pcb_tables(runs_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    for metrics_path in sorted(runs_root.glob("**/metrics_summary.json")):
        run_dir = metrics_path.parent
        manifest_path = resolve_manifest_path(run_dir)
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_section = manifest.get("eval") or manifest.get("train") or manifest.get("distill") or {}
        data_path = str(config_section.get("data") or config_section.get("prepared_data") or "")
        dataset_name = infer_dataset_name(data_path)
        if not dataset_name:
            continue
        experiment_name = str(config_section.get("name", run_dir.name))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_path = ROOT / str(config_section.get("model") or config_section.get("student_model") or "")
        model_stats = compute_model_stats(model_path) if model_path.exists() else {"params_m": "", "flops_b": ""}
        row = {
            "dataset": dataset_name,
            "experiment": experiment_name,
            "params_m": model_stats["params_m"],
            "flops_b": model_stats["flops_b"],
            "fps": speed_to_fps(metrics.get("speed")),
            "mAP50": extract_metric(metrics, "metrics/mAP50(B)"),
            "mAP50-95": extract_metric(metrics, "metrics/mAP50-95(B)"),
            "recall": extract_metric(metrics, "metrics/recall(B)"),
        }
        main_rows.append(row)
        if "epfa" in experiment_name.lower() or "plain" in experiment_name.lower():
            ablation_rows.append(row)

    write_csv(output_dir / "pcb_main_results.csv", main_rows)
    write_csv(output_dir / "pcb_ablation.csv", ablation_rows or main_rows)


def resolve_manifest_path(run_dir: Path) -> Path:
    direct = run_dir / "run_manifest.json"
    if direct.exists():
        return direct

    name = run_dir.name
    suffixes = ("_eval2", "_eval")
    for suffix in suffixes:
        if name.endswith(suffix):
            train_dir = run_dir.with_name(name[: -len(suffix)])
            candidate = train_dir / "run_manifest.json"
            if candidate.exists():
                return candidate

    return direct


def infer_dataset_name(data_path: str) -> str:
    normalized = data_path.lower()
    for dataset in ("deeppcb", "pku_market_pcb", "dspcbsd_plus"):
        if dataset in normalized:
            return dataset
    return ""


def compute_model_stats(model_path: Path) -> dict[str, str]:
    try:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops
    except Exception:
        return {"params_m": "", "flops_b": ""}

    register_ultralytics_modules()
    model = YOLO(str(model_path))
    initialize_custom_model_context(model.model)
    params = sum(parameter.numel() for parameter in model.model.parameters())
    flops = float(get_flops(model.model, imgsz=640))
    flops_text = f"{flops:.3f}" if flops > 0 else ""
    return {"params_m": f"{params / 1e6:.3f}", "flops_b": flops_text}


def speed_to_fps(speed) -> str:
    if not isinstance(speed, dict):
        return ""
    inference_ms = speed.get("inference")
    if not isinstance(inference_ms, (int, float)) or inference_ms <= 0:
        return ""
    return f"{1000.0 / float(inference_ms):.2f}"


def extract_metric(metrics: dict[str, object], key: str) -> str:
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
