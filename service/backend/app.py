from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from src.yolodist.paths import ROOT

from .infer import InferEngine
from .model_registry import ModelRegistry


DB_PATH = ROOT / "service" / "backend" / "data" / "inference_logs.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path.as_posix()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT,
                image_path TEXT,
                latency_ms REAL,
                detections_count INTEGER,
                status TEXT NOT NULL,
                error TEXT
            )
            """
        )
        conn.commit()


def log_inference(
    db_path: Path,
    *,
    model_name: str | None,
    image_path: str | None,
    latency_ms: float | None,
    detections_count: int | None,
    status: str,
    error: str | None = None,
) -> None:
    with sqlite3.connect(db_path.as_posix()) as conn:
        conn.execute(
            """
            INSERT INTO inference_logs (
                timestamp, model_name, image_path, latency_ms, detections_count, status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                model_name,
                image_path,
                latency_ms,
                detections_count,
                status,
                error,
            ),
        )
        conn.commit()


def build_stats_summary(db_path: Path) -> Dict[str, Any]:
    with sqlite3.connect(db_path.as_posix()) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) AS c FROM inference_logs").fetchone()["c"]
        success = conn.execute(
            "SELECT COUNT(*) AS c FROM inference_logs WHERE status='success'"
        ).fetchone()["c"]
        failed = conn.execute(
            "SELECT COUNT(*) AS c FROM inference_logs WHERE status='error'"
        ).fetchone()["c"]
        avg_latency_row = conn.execute(
            "SELECT AVG(latency_ms) AS avg_latency FROM inference_logs WHERE status='success'"
        ).fetchone()
        by_model_rows = conn.execute(
            """
            SELECT
                COALESCE(model_name, 'unknown') AS model_name,
                COUNT(*) AS total_requests,
                AVG(latency_ms) AS avg_latency_ms
            FROM inference_logs
            GROUP BY COALESCE(model_name, 'unknown')
            ORDER BY total_requests DESC
            """
        ).fetchall()

    return {
        "total_requests": int(total),
        "success_requests": int(success),
        "failed_requests": int(failed),
        "avg_latency_ms": (
            round(float(avg_latency_row["avg_latency"]), 3)
            if avg_latency_row["avg_latency"] is not None
            else None
        ),
        "by_model": [
            {
                "model_name": row["model_name"],
                "total_requests": int(row["total_requests"]),
                "avg_latency_ms": (
                    round(float(row["avg_latency_ms"]), 3)
                    if row["avg_latency_ms"] is not None
                    else None
                ),
            }
            for row in by_model_rows
        ],
    }


def create_app() -> Flask:
    init_db(DB_PATH)
    app = Flask(__name__, template_folder="templates")
    registry = ModelRegistry()
    infer_engine = InferEngine(registry=registry)

    @app.get("/api/v1/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "time_utc": utc_now_iso(),
                "current_model": registry.current_model(),
                "db_path": DB_PATH.as_posix(),
            }
        )

    @app.get("/")
    def index() -> Any:
        return render_template("index.html")

    @app.get("/api/v1/models")
    def models() -> Any:
        return jsonify(registry.as_json())

    @app.post("/api/v1/models/switch")
    def switch_model() -> Any:
        payload = request.get_json(silent=True) or {}
        model_name = payload.get("model_name")
        if not model_name:
            return jsonify({"error": "model_name is required"}), 400
        try:
            selected_path = registry.switch(str(model_name))
            return jsonify(
                {
                    "ok": True,
                    "current_model": registry.current_model(),
                    "weights_path": selected_path.as_posix(),
                }
            )
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/v1/infer")
    def infer() -> Any:
        payload = request.get_json(silent=True) or {}
        image_path = payload.get("image_path")
        model_name = payload.get("model_name")
        conf = payload.get("conf")

        if not image_path:
            return jsonify({"error": "image_path is required"}), 400
        if conf is not None:
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                return jsonify({"error": "conf must be a float"}), 400

        try:
            result = infer_engine.infer(
                image_path=str(image_path),
                model_name=str(model_name) if model_name else None,
                conf=conf,
            )
            log_inference(
                DB_PATH,
                model_name=result["model_name"],
                image_path=str(image_path),
                latency_ms=float(result["latency_ms"]),
                detections_count=len(result["detections"]),
                status="success",
            )
            return jsonify(result)
        except Exception as exc:  # keep API stable for service consumers
            target_model = str(model_name) if model_name else registry.current_model()
            log_inference(
                DB_PATH,
                model_name=target_model,
                image_path=str(image_path),
                latency_ms=None,
                detections_count=None,
                status="error",
                error=str(exc),
            )
            if isinstance(exc, FileNotFoundError):
                return jsonify({"error": str(exc)}), 404
            if isinstance(exc, KeyError):
                return jsonify({"error": str(exc)}), 400
            if isinstance(exc, RuntimeError):
                return jsonify({"error": str(exc)}), 503
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/v1/stats/summary")
    def stats_summary() -> Any:
        return jsonify(build_stats_summary(DB_PATH))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
