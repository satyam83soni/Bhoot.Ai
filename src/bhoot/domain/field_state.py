"""Per-field state machine.

A field moves PENDING → CANDIDATE → VALIDATED → VERIFIED (or FAILED_CHECKS / UNGROUNDED →
repair → CANDIDATE …) until the policy marks it final: ACCEPT, REVIEW or ABSENT. "Verified"
means grounded on the page **and** every applicable rule passed; the table below makes any
other route to VERIFIED impossible.

``FieldState`` is immutable: every change returns a new instance, so graph checkpoints never
see an object that was modified after it was stored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from bhoot.domain.candidate import CandidateRole, CandidateValue
from bhoot.domain.checks import CheckResult
from bhoot.domain.decisions import DecisionKind
from bhoot.domain.errors import IllegalTransitionError
from bhoot.domain.modes import Tier
from bhoot.domain.schema import FieldPath, FieldSpec
from bhoot.domain.work import WorkKind


class FieldStatus(StrEnum):
    PENDING = "pending"  # no candidate yet
    CANDIDATE = "candidate"  # has a value, not yet checked
    VALIDATED = "validated"  # passed type and rule checks
    VERIFIED = "verified"  # validated + grounded on the page
    FAILED_CHECKS = "failed_checks"  # a type or rule check failed
    UNGROUNDED = "ungrounded"  # quote not found, or value not inside the quote
    NEEDS_REVIEW = "needs_review"  # the policy gave up: a human must look
    ABSENT = "absent"  # optional field legitimately not in the document


S = FieldStatus
ALLOWED_TRANSITIONS: Mapping[FieldStatus, frozenset[FieldStatus]] = MappingProxyType({
    S.PENDING: frozenset({S.CANDIDATE, S.FAILED_CHECKS, S.NEEDS_REVIEW, S.ABSENT}),
    S.CANDIDATE: frozenset({S.CANDIDATE, S.VALIDATED, S.FAILED_CHECKS, S.NEEDS_REVIEW, S.ABSENT}),
    S.VALIDATED: frozenset({S.CANDIDATE, S.VERIFIED, S.FAILED_CHECKS, S.UNGROUNDED, S.NEEDS_REVIEW}),
    S.FAILED_CHECKS: frozenset({S.CANDIDATE, S.NEEDS_REVIEW}),
    S.UNGROUNDED: frozenset({S.CANDIDATE, S.NEEDS_REVIEW}),
    S.VERIFIED: frozenset({S.CANDIDATE, S.FAILED_CHECKS, S.NEEDS_REVIEW}),
    S.NEEDS_REVIEW: frozenset(),
    S.ABSENT: frozenset({S.CANDIDATE}),
})  # fmt: skip


@dataclass(frozen=True, slots=True)
class Attempt:
    tier: Tier
    model: str | None
    work_item_id: str
    kind: WorkKind | None  # None for Tier 0
    round: int
    outcome: FieldStatus  # updated as the field moves on from this attempt
    reason: str | None = None
    cost_usd: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class FieldState:
    path: FieldPath  # concrete
    spec: FieldSpec
    status: FieldStatus = FieldStatus.PENDING
    current: CandidateValue | None = None
    alternatives: tuple[CandidateValue, ...] = ()
    attempts: tuple[Attempt, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    confidence: float = 0.0
    confidence_signals: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    cross_checked: bool = False
    crop_checked: bool = False
    final: DecisionKind | None = None
    decision_log: tuple[str, ...] = ()
    history: tuple[tuple[FieldStatus, FieldStatus, str | None], ...] = ()

    # ---------------------------------------------------------------- queries
    @property
    def is_terminal(self) -> bool:
        return self.final is not None

    @property
    def value(self) -> Any:
        return self.current.value if self.current else None

    def can_transition(self, to: FieldStatus) -> bool:
        return to in ALLOWED_TRANSITIONS[self.status]

    # ---------------------------------------------------------------- changes
    def transition(self, to: FieldStatus, reason: str | None = None, **changes: Any) -> FieldState:
        """Move to ``to`` (raises IllegalTransitionError if the table forbids it)."""
        if not self.can_transition(to):
            raise IllegalTransitionError(
                f"{self.path}: {self.status.value} -> {to.value} is not allowed ({reason or 'no reason'})"
            )
        attempts = self.attempts
        if attempts and to is not FieldStatus.CANDIDATE:
            attempts = (*attempts[:-1], replace(attempts[-1], outcome=to, reason=reason))
        # any status change re-opens a closed field unless a final decision comes with it
        final = changes.pop("final", None)
        return replace(
            self,
            status=to,
            attempts=attempts,
            final=final,
            history=(*self.history, (self.status, to, reason)),
            **changes,
        )

    def with_candidate(self, candidate: CandidateValue, reason: str | None = None) -> FieldState:
        """A new primary value arrives: the field becomes CANDIDATE and its old checks are cleared."""
        if candidate.role is not CandidateRole.PRIMARY:
            raise ValueError("alternatives are added with with_alternative()")
        attempt = Attempt(
            tier=candidate.tier,
            model=candidate.model,
            work_item_id=candidate.work_item_id,
            kind=candidate.source_kind,
            round=candidate.round,
            outcome=FieldStatus.CANDIDATE,
        )
        crop_checked = self.crop_checked or candidate.source_kind is WorkKind.CROP
        return replace(self, attempts=(*self.attempts, attempt)).transition(
            FieldStatus.CANDIDATE,
            reason or f"value from {candidate.work_item_id}",
            current=candidate,
            checks=(),
            crop_checked=crop_checked,
        )

    def with_alternative(self, candidate: CandidateValue) -> FieldState:
        """A cross-check answer is stored next to the value; status does not change."""
        return replace(self, alternatives=(*self.alternatives, candidate))

    def decide(self, kind: DecisionKind, reason: str, rule: str, round_: int) -> FieldState:
        """Record a policy decision; final kinds (ACCEPT / REVIEW / ABSENT) close the field."""
        log = (*self.decision_log, f"r{round_}:{rule}:{kind.value}")
        if kind is DecisionKind.REVIEW:
            return self.transition(FieldStatus.NEEDS_REVIEW, reason, final=kind, decision_log=log)
        if kind is DecisionKind.ABSENT:
            if self.status is FieldStatus.ABSENT:
                return replace(self, final=kind, decision_log=log)
            return self.transition(FieldStatus.ABSENT, reason, final=kind, decision_log=log)
        if kind is DecisionKind.ACCEPT:
            if self.status is not FieldStatus.VERIFIED:
                raise IllegalTransitionError(f"{self.path}: only VERIFIED fields can be accepted")
            return replace(self, final=kind, decision_log=log)
        return replace(self, decision_log=log)
