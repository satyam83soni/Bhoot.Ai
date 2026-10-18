"""What the escalation policy decided for a field in one round."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bhoot.domain.modes import Tier
from bhoot.domain.schema import FieldPath


class DecisionKind(StrEnum):
    ACCEPT = "accept"
    REPAIR = "repair"
    ESCALATE = "escalate"
    CROSS_CHECK = "cross_check"
    REVIEW = "review"
    ABSENT = "absent"

    @property
    def is_final(self) -> bool:
        return self in (DecisionKind.ACCEPT, DecisionKind.REVIEW, DecisionKind.ABSENT)


@dataclass(frozen=True, slots=True)
class Decision:
    path: FieldPath
    kind: DecisionKind
    reason: str
    rule: str  # policy rule or guard that produced it ("rule8", "G1"), for explanations
    target_tier: Tier | None = None
