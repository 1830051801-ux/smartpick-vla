"""Task contracts and leakage-resistant instruction splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from smartpick_vla.data.schema import QualityClass

InstructionSplit = Literal["train", "paraphrase", "ood"]


INSTRUCTION_TEMPLATES: dict[InstructionSplit, dict[QualityClass, tuple[str, ...]]] = {
    "train": {
        "accepted": (
            "place the accepted part in the accepted tray",
            "sort the good component into the green bin",
            "pick the approved item and put it in the accepted bin",
            "move the pass-quality workpiece to its tray",
        ),
        "scratch": (
            "place the scratched part in the scratch tray",
            "sort the surface-defect component into the orange bin",
            "pick the scratched item and put it in the scratch bin",
            "move the marked workpiece to the defect tray",
        ),
        "unknown": (
            "place the unknown part in the inspection tray",
            "sort the uncertain component into the purple bin",
            "pick the unclassified item and put it in the unknown bin",
            "move the ambiguous workpiece to manual inspection",
        ),
    },
    "paraphrase": {
        "accepted": (
            "route the quality-approved piece to the pass container",
            "send the conforming unit to the acceptance tote",
        ),
        "scratch": (
            "route the cosmetically damaged piece to the blemish container",
            "send the scored unit to the surface-rework tote",
        ),
        "unknown": (
            "route the undecided piece to the review container",
            "send the indeterminate unit for human inspection",
        ),
    },
    "ood": {
        "accepted": (
            "the conforming article belongs with passed goods",
            "transfer the no-defect sample to acceptance",
        ),
        "scratch": (
            "the visibly marred article belongs with rework goods",
            "transfer the abrasion sample to cosmetic rejection",
        ),
        "unknown": (
            "the unresolved article needs a human decision",
            "transfer the low-confidence sample to manual review",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class SortTask:
    target_class: QualityClass
    instruction: str
    template_id: str
    split: InstructionSplit


def sample_task(
    rng: np.random.Generator,
    *,
    split: InstructionSplit = "train",
    target_class: QualityClass | None = None,
) -> SortTask:
    """Sample a class-balanced language-conditioned sorting task."""

    classes: tuple[QualityClass, ...] = ("accepted", "scratch", "unknown")
    selected_class = target_class or classes[int(rng.integers(0, len(classes)))]
    templates = INSTRUCTION_TEMPLATES[split][selected_class]
    template_index = int(rng.integers(0, len(templates)))
    return SortTask(
        target_class=selected_class,
        instruction=templates[template_index],
        template_id=f"{split}/{selected_class}/{template_index}",
        split=split,
    )


def all_instructions(split: InstructionSplit | None = None) -> tuple[str, ...]:
    splits = (split,) if split is not None else tuple(INSTRUCTION_TEMPLATES)
    categories: tuple[QualityClass, ...] = ("accepted", "scratch", "unknown")
    return tuple(
        instruction
        for split_name in splits
        for category in categories
        for instruction in INSTRUCTION_TEMPLATES[split_name][category]
    )
