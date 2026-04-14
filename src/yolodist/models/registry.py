from __future__ import annotations

from yolodist.models.ppla import PPLALite


def register_ultralytics_modules() -> None:
    """Register custom YOLODist modules into Ultralytics parse namespace."""
    try:
        import ultralytics.nn.tasks as tasks
    except Exception:
        return

    # Ultralytics parse_model resolves custom layers from tasks module globals.
    if getattr(tasks, "PPLALite", None) is None:
        setattr(tasks, "PPLALite", PPLALite)

