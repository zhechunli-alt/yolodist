from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json


@dataclass
class RunSummary:
    run_dir: Path
    best_epoch: str
    metrics: dict[str, str]


def summarize_run(run_dir: Path) -> RunSummary:
    results_csv = run_dir / "results.csv"
    metrics_json = run_dir / "metrics_summary.json"

    if results_csv.exists():
        best_epoch, metrics = summarize_results_csv(results_csv)
        return RunSummary(run_dir=run_dir, best_epoch=best_epoch, metrics=metrics)

    if metrics_json.exists():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        metrics = {str(k): str(v) for k, v in data.items()}
        return RunSummary(run_dir=run_dir, best_epoch="n/a", metrics=metrics)

    raise FileNotFoundError(f"No supported result file found in {run_dir}")


def summarize_results_csv(results_csv: Path) -> tuple[str, dict[str, str]]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"results.csv is empty: {results_csv}")

    best_row = choose_best_row(rows)
    best_epoch = best_row.get("epoch", "unknown")
    metrics = {
        key: value
        for key, value in best_row.items()
        if key.startswith("metrics/") or key.startswith("val/") or key in {"fitness", "lr/pg0", "lr/pg1", "lr/pg2"}
    }
    return str(best_epoch), metrics


def choose_best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    preferred_keys = ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "fitness")
    for key in preferred_keys:
        numeric_rows = [row for row in rows if parse_float(row.get(key)) is not None]
        if numeric_rows:
            return max(numeric_rows, key=lambda row: parse_float(row.get(key)) or float("-inf"))
    return rows[-1]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_markdown(summary: RunSummary) -> str:
    lines = [
        f"# Run Summary: {summary.run_dir.name}",
        "",
        f"- Run directory: `{summary.run_dir}`",
        f"- Best epoch: `{summary.best_epoch}`",
        "",
        "## Metrics",
    ]
    for key in sorted(summary.metrics):
        lines.append(f"- `{key}`: `{summary.metrics[key]}`")
    lines.append("")
    return "\n".join(lines)
