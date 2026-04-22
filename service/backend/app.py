from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from uuid import uuid4

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src.yolodist.paths import ROOT

from .infer import InferEngine
from .model_registry import ModelRegistry


DB_PATH = ROOT / "service" / "backend" / "data" / "inference_logs.db"
UPLOAD_DIR = ROOT / "service" / "backend" / "data" / "uploads"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_uuid TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                sample_name TEXT,
                sample_path TEXT,
                image_width INTEGER,
                image_height INTEGER,
                model_name TEXT,
                latency_ms REAL,
                detections_count INTEGER,
                max_conf REAL,
                avg_conf REAL,
                meta_json TEXT,
                result_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_uuid TEXT NOT NULL,
                det_index INTEGER NOT NULL,
                cls INTEGER,
                class_name TEXT,
                conf REAL,
                x1 REAL,
                y1 REAL,
                x2 REAL,
                y2 REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_uuid TEXT NOT NULL,
                doctor_name TEXT,
                diagnosis TEXT,
                risk_level TEXT,
                created_at TEXT NOT NULL
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


def _error_response_from_exception(exc: Exception) -> Tuple[Response, int]:
    if isinstance(exc, FileNotFoundError):
        return jsonify({"error": str(exc)}), 404
    if isinstance(exc, KeyError):
        return jsonify({"error": str(exc)}), 400
    if isinstance(exc, RuntimeError):
        return jsonify({"error": str(exc)}), 503
    return jsonify({"error": str(exc)}), 500


def _safe_json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def create_app() -> Flask:
    init_db(DB_PATH)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, template_folder="templates")
    registry = ModelRegistry()
    infer_engine = InferEngine(registry=registry)

    def save_uploaded_image(image_file: Any) -> Path:
        if image_file is None or image_file.filename is None:
            raise ValueError("image file is required")
        suffix = Path(image_file.filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(
                f"unsupported image format: {suffix or '(none)'}; allowed: {sorted(ALLOWED_IMAGE_SUFFIXES)}"
            )
        safe_name = secure_filename(image_file.filename) or f"upload{suffix}"
        dst_path = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"
        image_file.save(dst_path.as_posix())
        return dst_path

    def run_infer(
        *,
        image_path: str,
        model_name: str | None,
        conf: float | None,
    ) -> Dict[str, Any]:
        result = infer_engine.infer(
            image_path=image_path,
            model_name=model_name if model_name else None,
            conf=conf,
        )
        log_inference(
            DB_PATH,
            model_name=result.get("model_name"),
            image_path=image_path,
            latency_ms=float(result.get("latency_ms", 0.0)),
            detections_count=len(result.get("detections", [])),
            status="success",
        )
        return result

    @app.get("/")
    def index() -> Any:
        return render_template("index.html")

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
        except Exception as exc:
            return _error_response_from_exception(exc)

    @app.post("/api/v1/infer")
    def infer() -> Any:
        payload = request.get_json(silent=True) or {}
        image_path = payload.get("image_path")
        model_name = payload.get("model_name")
        conf_raw = payload.get("conf")

        if not image_path:
            return jsonify({"error": "image_path is required"}), 400
        conf = None
        if conf_raw is not None:
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "conf must be a float"}), 400

        try:
            return jsonify(
                run_infer(
                    image_path=str(image_path),
                    model_name=str(model_name) if model_name else None,
                    conf=conf,
                )
            )
        except Exception as exc:
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
            return _error_response_from_exception(exc)

    @app.post("/api/v1/infer/upload")
    def infer_upload() -> Any:
        image_file = request.files.get("image")
        model_name = request.form.get("model_name")
        conf_raw = request.form.get("conf")
        conf = None
        if conf_raw:
            try:
                conf = float(conf_raw)
            except ValueError:
                return jsonify({"error": "conf must be a float"}), 400

        try:
            dst_path = save_uploaded_image(image_file)
            result = run_infer(
                image_path=dst_path.as_posix(),
                model_name=model_name if model_name else None,
                conf=conf,
            )
            result["uploaded_image_path"] = dst_path.as_posix()
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            log_inference(
                DB_PATH,
                model_name=model_name if model_name else registry.current_model(),
                image_path=None,
                latency_ms=None,
                detections_count=None,
                status="error",
                error=str(exc),
            )
            return _error_response_from_exception(exc)

    @app.post("/api/v1/infer/compare")
    def infer_compare() -> Any:
        image_file = request.files.get("image")
        conf_raw = request.form.get("conf")
        model_names_raw = request.form.get("model_names", "")

        conf = None
        if conf_raw:
            try:
                conf = float(conf_raw)
            except ValueError:
                return jsonify({"error": "conf must be a float"}), 400

        try:
            dst_path = save_uploaded_image(image_file)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        available_models = [m.name for m in registry.list_models() if m.exists]
        if not available_models:
            return jsonify({"error": "no available model in weights/"}), 503

        requested_models: List[str]
        if not model_names_raw.strip():
            requested_models = available_models
        else:
            requested_models = [x.strip() for x in model_names_raw.split(",") if x.strip()]
            requested_models = [m for m in requested_models if m in available_models]
            if not requested_models:
                return jsonify({"error": "no valid model_names provided"}), 400

        results: List[Dict[str, Any]] = []
        for name in requested_models:
            try:
                out = run_infer(image_path=dst_path.as_posix(), model_name=name, conf=conf)
                results.append(out)
            except Exception as exc:  # per-model failure should not block others
                results.append({"model_name": name, "error": str(exc)})

        return jsonify(
            {
                "uploaded_image_path": dst_path.as_posix(),
                "results": results,
                "model_names": requested_models,
            }
        )

    @app.post("/api/v1/records/save")
    def save_record() -> Any:
        payload = request.get_json(silent=True) or {}
        sample_meta = payload.get("sample_meta") or {}
        infer_result = payload.get("infer_result") or {}
        doctor_note = payload.get("doctor_note") or {}

        if not infer_result.get("model_name"):
            return jsonify({"error": "infer_result.model_name is required"}), 400

        detections = infer_result.get("detections") or []
        confs = [float(x.get("conf", 0.0)) for x in detections]
        max_conf = max(confs) if confs else None
        avg_conf = (sum(confs) / len(confs)) if confs else None

        image_size = infer_result.get("image_size") or {}
        record_uuid = uuid4().hex
        created_at = utc_now_iso()

        with sqlite3.connect(DB_PATH.as_posix()) as conn:
            conn.execute(
                """
                INSERT INTO inspection_records (
                    record_uuid, created_at, sample_name, sample_path, image_width, image_height,
                    model_name, latency_ms, detections_count, max_conf, avg_conf, meta_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_uuid,
                    created_at,
                    str(sample_meta.get("sample_name", "")),
                    str(sample_meta.get("sample_path", "")),
                    int(image_size.get("width", 0) or 0),
                    int(image_size.get("height", 0) or 0),
                    str(infer_result.get("model_name", "")),
                    float(infer_result.get("latency_ms", 0.0) or 0.0),
                    int(len(detections)),
                    float(max_conf) if max_conf is not None else None,
                    float(avg_conf) if avg_conf is not None else None,
                    _safe_json_dump(sample_meta),
                    _safe_json_dump(infer_result),
                ),
            )
            for idx, det in enumerate(detections):
                xyxy = det.get("xyxy") or [None, None, None, None]
                conn.execute(
                    """
                    INSERT INTO detection_items (
                        record_uuid, det_index, cls, class_name, conf, x1, y1, x2, y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_uuid,
                        idx,
                        int(det.get("cls", -1)),
                        str(det.get("class_name", "")),
                        float(det.get("conf", 0.0)),
                        float(xyxy[0]) if xyxy[0] is not None else None,
                        float(xyxy[1]) if xyxy[1] is not None else None,
                        float(xyxy[2]) if xyxy[2] is not None else None,
                        float(xyxy[3]) if xyxy[3] is not None else None,
                    ),
                )
            conn.execute(
                """
                INSERT INTO review_notes (
                    record_uuid, doctor_name, diagnosis, risk_level, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record_uuid,
                    str(doctor_note.get("doctor_name", "")),
                    str(doctor_note.get("diagnosis", "")),
                    str(doctor_note.get("risk_level", "")),
                    created_at,
                ),
            )
            conn.commit()

        return jsonify({"ok": True, "record_uuid": record_uuid})

    @app.get("/api/v1/records/list")
    def list_records() -> Any:
        limit_raw = request.args.get("limit", "30")
        try:
            limit = max(1, min(200, int(limit_raw)))
        except ValueError:
            return jsonify({"error": "limit must be integer"}), 400

        with sqlite3.connect(DB_PATH.as_posix()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    r.record_uuid,
                    r.created_at,
                    r.sample_name,
                    r.sample_path,
                    r.model_name,
                    r.latency_ms,
                    r.detections_count,
                    r.max_conf,
                    n.doctor_name,
                    n.risk_level
                FROM inspection_records r
                LEFT JOIN review_notes n ON n.record_uuid = r.record_uuid
                ORDER BY r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return jsonify({"records": [dict(r) for r in rows]})

    @app.get("/api/v1/report/export")
    def export_report() -> Any:
        record_uuid = request.args.get("record_uuid", "").strip()
        if not record_uuid:
            return jsonify({"error": "record_uuid is required"}), 400

        with sqlite3.connect(DB_PATH.as_posix()) as conn:
            conn.row_factory = sqlite3.Row
            record = conn.execute(
                "SELECT * FROM inspection_records WHERE record_uuid = ?",
                (record_uuid,),
            ).fetchone()
            if record is None:
                return jsonify({"error": "record not found"}), 404
            items = conn.execute(
                """
                SELECT det_index, cls, class_name, conf, x1, y1, x2, y2
                FROM detection_items
                WHERE record_uuid = ?
                ORDER BY det_index ASC
                """,
                (record_uuid,),
            ).fetchall()
            note = conn.execute(
                """
                SELECT doctor_name, diagnosis, risk_level, created_at
                FROM review_notes
                WHERE record_uuid = ?
                ORDER BY id DESC LIMIT 1
                """,
                (record_uuid,),
            ).fetchone()

        report = {
            "record": dict(record),
            "detection_items": [dict(x) for x in items],
            "review_note": dict(note) if note else None,
            "export_time": utc_now_iso(),
        }
        content = json.dumps(report, ensure_ascii=False, indent=2)
        return Response(
            content,
            mimetype="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=inspection_report_{record_uuid}.json"
            },
        )

    @app.get("/api/v1/stats/summary")
    def stats_summary() -> Any:
        return jsonify(build_stats_summary(DB_PATH))

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "5000"))
    app.run(host=host, port=port, debug=False)

