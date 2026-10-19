"""The extraction graph's state and its reducers.

Only plain, serialisable data lives here (no packs, no callables, no secrets): every key is
checkpointed after each superstep. Keys written by parallel ``llm_extract`` branches have
reducers; every other key has a single writer per step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from bhoot.domain.candidate import CandidateValue
from bhoot.domain.checks import CheckResult
from bhoot.domain.cost import CostReport
from bhoot.domain.decisions import Decision
from bhoot.domain.document import Document
from bhoot.domain.field_state import FieldState
from bhoot.domain.schema import FieldPath
from bhoot.domain.work import WorkItem, WorkOutcome
from bhoot.engine.normalize.conventions import DocConventions


@dataclass(frozen=True, slots=True)
class Reset:
    """Written to an appending key to replace its contents (``Reset()`` empties it)."""

    items: tuple[Any, ...] = ()


def append_or_reset(old: tuple[Any, ...], new: tuple[Any, ...] | Reset) -> tuple[Any, ...]:
    if isinstance(new, Reset):
        return tuple(new.items)
    return (*old, *new)


def merge_dict(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    return {**old, **new}


def concat(old: tuple[Any, ...], new: tuple[Any, ...]) -> tuple[Any, ...]:
    return (*old, *new)


def cost_merge(old: CostReport, new: CostReport) -> CostReport:
    return CostReport.merge(old, new)


class ExtractionState(TypedDict, total=False):
    job_id: str
    document: Document
    conventions: DocConventions | None
    section_pages: dict[str, tuple[int, ...]]
    work_items: tuple[WorkItem, ...]
    work_outcomes: Annotated[dict[str, WorkOutcome], merge_dict]
    candidates: Annotated[tuple[CandidateValue, ...], append_or_reset]
    nulls: Annotated[tuple[FieldPath, ...], append_or_reset]
    returned_null: tuple[str, ...]
    field_states: dict[str, FieldState]
    doc_checks: tuple[CheckResult, ...]
    decisions: tuple[Decision, ...]
    round: int
    status: str
    cost: Annotated[CostReport, cost_merge]
    timings_ms: Annotated[dict[str, float], merge_dict]
    warnings: Annotated[tuple[str, ...], concat]
