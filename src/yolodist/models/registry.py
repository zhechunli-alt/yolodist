from __future__ import annotations

from yolodist.models.epfa import EPFALite
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
    if getattr(tasks, "EPFALite", None) is None:
        setattr(tasks, "EPFALite", EPFALite)


def initialize_custom_model_context(model) -> None:
    if getattr(model, "_yolodist_context_hook", None) is not None:
        return

    def _set_input(_, args):
        image = args[0] if args else None
        EPFALite.set_current_input(image)

    model._yolodist_context_hook = model.register_forward_pre_hook(_set_input)
