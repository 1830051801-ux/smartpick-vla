"""Small transparent task parser for simulator language templates.

This is intentionally not presented as a large language model. It makes the
task-selection boundary auditable while the learned component focuses on
visual localization.
"""

from __future__ import annotations

import re

from smartpick_vla.data.schema import QualityClass

_CLASS_KEYWORDS: dict[QualityClass, tuple[str, ...]] = {
    "accepted": (
        "accepted",
        "approved",
        "good component",
        "pass-quality",
        "quality-approved",
        "conforming",
        "no-defect",
        "passed goods",
        "acceptance",
    ),
    "scratch": (
        "scratch",
        "scratched",
        "surface-defect",
        "marked workpiece",
        "damaged",
        "blemish",
        "scored",
        "rework",
        "marred",
        "abrasion",
        "cosmetic rejection",
    ),
    "unknown": (
        "unknown",
        "uncertain",
        "unclassified",
        "ambiguous",
        "inspection",
        "undecided",
        "indeterminate",
        "manual review",
        "human decision",
        "unresolved",
    ),
}


def infer_quality_class(instruction: str) -> QualityClass:
    """Resolve the requested quality class from a declared task template."""

    normalized = instruction.lower().strip()
    if not normalized:
        raise ValueError("instruction must be non-empty")
    scores = {
        category: sum(
            re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", normalized) is not None
            for keyword in keywords
        )
        for category, keywords in _CLASS_KEYWORDS.items()
    }
    best_score = max(scores.values())
    best = [category for category, score in scores.items() if score == best_score]
    if best_score < 1 or len(best) != 1:
        raise ValueError(f"cannot resolve a unique quality class from {instruction!r}")
    return best[0]
