"""Units of LLM work: what the engine asks a model to do, and how each request ended."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from bhoot.domain.modes import Tier
from bhoot.domain.schema import FieldPath


class WorkKind(StrEnum):
    EXTRACT = "extract"  # first extraction of a section
    REPAIR = "repair"  # re-ask failing fields with feedback
    CROP = "crop"  # re-read rows/fields from image crops (OCR pages)
    CROSSCHECK = "crosscheck"  # independent answer from a second model


class WorkStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    SKIPPED_BUDGET = "skipped_budget"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class WorkItem:
    """One LLM call: a section of the document, the fields wanted and how to ask for them.

    ``paths`` empty means every field of the section (first extraction); repairs and
    cross-checks list exactly the fields they target. ``row_window`` limits a row array to
    rows ``[start, end)`` so a repair answer replaces only that window.
    """

    item_id: str
    round: int
    kind: WorkKind
    section: str
    page_indices: tuple[int, ...]  # 0-based
    tier: Tier
    model_role: str  # cheap | strong | cross | vision_cheap
    paths: tuple[FieldPath, ...] = ()
    use_images: bool = False
    feedback: tuple[str, ...] = ()  # failed checks and hints shown to the model on repair
    crops: tuple[str, ...] = ()  # blob keys of image crops (CROP items)
    row_window: tuple[int, int] | None = None
    status: WorkStatus = WorkStatus.PENDING

    @staticmethod
    def make_id(round_: int, kind: WorkKind, section: str, pages: tuple[int, ...], n: int) -> str:
        span = f"p{pages[0] + 1}-{pages[-1] + 1}" if pages else "p0"
        return f"r{round_}-{section}-{span}-{kind.value}-{n}"

    def with_status(self, status: WorkStatus) -> WorkItem:
        return replace(self, status=status)


@dataclass(frozen=True, slots=True)
class WorkOutcome:
    item_id: str
    status: WorkStatus
    model: str | None = None
    error: str | None = None  # short message for explain() and repair feedback
