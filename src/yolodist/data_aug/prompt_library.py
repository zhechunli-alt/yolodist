from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_SEVERITIES = ("minor", "moderate", "severe")


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    category: str
    defect_type: str
    severity: str
    prompt: str


class MVTecPromptLibrary:
    """Build text prompts for MVTec defect synthesis in a reproducible way."""

    def __init__(self, style_hint: str = "industrial inspection photo") -> None:
        self.style_hint = style_hint.strip()

    def build(
        self,
        category: str,
        defect_type: str,
        severity: str,
        sample_index: int,
    ) -> PromptTemplate:
        clean_category = category.strip().lower().replace(" ", "_")
        clean_defect = defect_type.strip().lower().replace(" ", "_")
        clean_severity = severity.strip().lower().replace(" ", "_")
        prompt_id = f"{clean_category}-{clean_defect}-{clean_severity}-{sample_index:04d}"

        prompt = (
            f"{self.style_hint}, mvtec category {category}, "
            f"defect type {defect_type}, severity {severity}, "
            "single object centered, high detail, realistic texture, no text overlay"
        )
        return PromptTemplate(
            prompt_id=prompt_id,
            category=category,
            defect_type=defect_type,
            severity=severity,
            prompt=prompt,
        )


def normalize_severities(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return list(DEFAULT_SEVERITIES)
    result = [value.strip().lower() for value in values if value and value.strip()]
    return result or list(DEFAULT_SEVERITIES)
