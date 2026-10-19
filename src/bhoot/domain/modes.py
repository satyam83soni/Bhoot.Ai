"""Extraction modes and the cost/accuracy tier ladder."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field

from bhoot.domain.schema import Criticality


class Mode(StrEnum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    MAX_ACCURACY = "max_accuracy"


class Tier(IntEnum):
    CODE = 0  # deterministic extraction, no LLM
    CHEAP = 1  # cheap model
    REPAIR = 2  # cheap model, targeted re-ask with feedback
    STRONG = 3  # strong model
    CROSS = 4  # independent second model for critical fields
    REVIEW = 5  # human review


class ModeProfile(BaseModel):
    """How far up the ladder each criticality may go, and the thresholds used on the way."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Mode
    max_tier_by_criticality: Mapping[Criticality, Tier]
    max_repairs_per_field: int = Field(ge=0)
    max_rounds: int = Field(ge=1)
    cross_check_threshold: float | None = Field(default=None, ge=0, le=1)  # None: never cross-check
    fuzzy_quote_threshold: float = Field(ge=0, le=100)
    allow_tier0_only_accept: bool = True

    def max_tier(self, criticality: Criticality) -> Tier:
        return self.max_tier_by_criticality[criticality]
