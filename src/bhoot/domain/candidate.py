"""Candidate values: what an extractor proposed for a field, before it is checked."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from bhoot.domain.evidence import Evidence
from bhoot.domain.modes import Tier
from bhoot.domain.schema import FieldPath
from bhoot.domain.work import WorkKind


class CandidateRole(StrEnum):
    PRIMARY = "primary"  # replaces the field's current value
    ALTERNATIVE = "alternative"  # independent second answer (cross-check); never replaces


@dataclass(frozen=True, slots=True)
class CandidateValue:
    path: FieldPath  # concrete path, e.g. transactions[12].balance
    raw: Any  # as produced, before coercion
    work_item_id: str
    round: int
    tier: Tier
    value: Any = None  # typed value after normalisation
    quote: str | None = None
    page: int | None = None  # 0-based
    model: str | None = None  # None for Tier 0
    role: CandidateRole = CandidateRole.PRIMARY
    source_kind: WorkKind | None = None  # None for Tier 0
    evidence: Evidence | None = None  # set by verification (or by Tier 0 directly)
    coercion_error: str | None = None

    @property
    def is_code(self) -> bool:
        return self.tier is Tier.CODE

    def typed(self, value: Any) -> CandidateValue:
        return replace(self, value=value, coercion_error=None)

    def failed_coercion(self, error: str) -> CandidateValue:
        return replace(self, value=None, coercion_error=error)

    def with_evidence(self, evidence: Evidence) -> CandidateValue:
        return replace(self, evidence=evidence)
