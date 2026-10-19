"""The escalation policy: what to do next with each field (LLD §9.15.2).

Stage 1 is an ordered list of rules (first match wins). Stage 2 guards wrap every decision,
so budget and round limits can never be skipped by an earlier rule. A guard never accepts a
field that isn't VERIFIED (grounded and every rule passed); anything else goes to review.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import PageKind
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.modes import ModeProfile, Tier
from bhoot.domain.schema import Criticality
from bhoot.engine.registry import StrategyRegistry

S = FieldStatus
D = DecisionKind
BROKEN = frozenset({S.FAILED_CHECKS, S.UNGROUNDED})


def _always(decision: Decision) -> bool:
    return True


@dataclass(frozen=True)
class FieldContext:
    state: FieldState
    profile: ModeProfile
    round: int
    page_kind: PageKind | None = None
    returned_null: bool = False  # the last call asking for this field answered null
    has_cross_model: bool = True
    alternatives_agree: bool | None = None  # None: no second answer compared
    critical_word_confidence: float = 0.90
    can_afford: Callable[[Decision], bool] = field(default=_always)

    @property
    def criticality(self) -> Criticality:
        return self.state.spec.criticality

    @property
    def max_tier(self) -> Tier:
        tier = self.profile.max_tier(self.criticality)
        if tier is Tier.CROSS and not self.has_cross_model:
            return Tier.STRONG  # no second model configured: cross-check is not available
        return tier

    @property
    def repairs(self) -> int:
        return sum(1 for line in self.state.decision_log if line.endswith(":repair"))

    @property
    def strong_tried(self) -> bool:
        return any(a.tier >= Tier.STRONG for a in self.state.attempts) or any(
            line.endswith(":escalate") for line in self.state.decision_log
        )

    @property
    def ocr_min_confidence(self) -> float | None:
        cand = self.state.current
        return cand.evidence.ocr_min_confidence if cand and cand.evidence else None

    def decision(
        self, kind: DecisionKind, reason: str, rule: str, tier: Tier | None = None
    ) -> Decision:
        return Decision(self.state.path, kind, reason, rule, tier)


class PolicyRule(Protocol):
    @property
    def name(self) -> str: ...

    def matches(self, f: FieldContext) -> bool: ...

    def decide(self, f: FieldContext) -> Decision: ...


@dataclass(frozen=True)
class _Rule:
    name: str
    when: Callable[[FieldContext], bool]
    then: Callable[[FieldContext], Decision]

    def matches(self, f: FieldContext) -> bool:
        return self.when(f)

    def decide(self, f: FieldContext) -> Decision:
        return self.then(f)


def _critical_verified(f: FieldContext) -> bool:
    return f.state.status is S.VERIFIED and f.criticality is Criticality.CRITICAL


RULES: list[PolicyRule] = [
    _Rule(
        "rule1",
        lambda f: _critical_verified(f) and f.state.cross_checked and f.alternatives_agree is False,
        lambda f: f.decision(D.REVIEW, "the two models disagree", "rule1"),
    ),
    _Rule(
        "rule2",
        lambda f: _critical_verified(f) and f.state.cross_checked and f.alternatives_agree is True,
        lambda f: f.decision(D.ACCEPT, "a second model agrees", "rule2"),
    ),
    _Rule(
        "rule3",
        lambda f: (
            f.state.status in (S.VALIDATED, S.VERIFIED)
            and f.page_kind is PageKind.OCR_TEXT
            and f.criticality is Criticality.CRITICAL
            and f.ocr_min_confidence is not None
            and f.ocr_min_confidence < f.critical_word_confidence
            and not f.state.crop_checked
        ),
        lambda f: f.decision(
            D.REPAIR, "low OCR confidence: re-read from the image", "rule3", Tier.REPAIR
        ),
    ),
    _Rule(
        "rule4",
        lambda f: (
            _critical_verified(f)
            and f.profile.cross_check_threshold is not None
            and f.state.confidence < f.profile.cross_check_threshold
            and f.max_tier >= Tier.CROSS
            and not f.state.cross_checked
        ),
        lambda f: f.decision(
            D.CROSS_CHECK,
            f"confidence {f.state.confidence:.2f} is low for a critical field",
            "rule4",
            Tier.CROSS,
        ),
    ),
    _Rule(
        "rule5",
        lambda f: f.state.status is S.VERIFIED,
        lambda f: f.decision(D.ACCEPT, "verified", "rule5"),
    ),
    _Rule(
        "rule6",
        lambda f: (
            f.state.status is S.PENDING
            and not (f.returned_null and not f.state.spec.required)
            and f.repairs < f.profile.max_repairs_per_field
        ),
        lambda f: f.decision(D.REPAIR, "no value yet", "rule6", Tier.REPAIR),
    ),
    _Rule(
        "rule7",
        lambda f: f.state.status is S.PENDING and f.returned_null and not f.state.spec.required,
        lambda f: f.decision(D.ABSENT, "not in the document", "rule7"),
    ),
    _Rule(
        "rule8",
        lambda f: (
            f.state.status in BROKEN
            and f.repairs < f.profile.max_repairs_per_field
            and f.max_tier >= Tier.REPAIR
        ),
        lambda f: f.decision(
            D.REPAIR, "checks failed: ask again with feedback", "rule8", Tier.REPAIR
        ),
    ),
    _Rule(
        "rule9",
        lambda f: (
            f.state.status in (*BROKEN, S.PENDING)
            and f.max_tier >= Tier.STRONG
            and not f.strong_tried
        ),
        lambda f: f.decision(
            D.ESCALATE, "repairs did not fix it: use the strong model", "rule9", Tier.STRONG
        ),
    ),
    _Rule(
        "rule10",
        lambda f: f.state.status in (*BROKEN, S.PENDING),
        lambda f: f.decision(D.REVIEW, "could not be verified within this mode's limits", "rule10"),
    ),
]

POLICY_RULES: StrategyRegistry[PolicyRule] = StrategyRegistry(
    "policy rule", {r.name: r for r in RULES}
)
DEFAULT_POLICY_RULES: tuple[str, ...] = tuple(r.name for r in RULES)


class EscalationPolicy:
    def __init__(self, rules: Sequence[PolicyRule]) -> None:
        self.rules = tuple(rules)

    def decide(self, f: FieldContext) -> Decision:
        for r in self.rules:
            if r.matches(f):
                return r.decide(f)
        return f.decision(D.REVIEW, f"no policy rule for status {f.state.status.value}", "fallback")


# ---------------------------------------------------------------- stage 2: guards
SPENDING = frozenset({D.REPAIR, D.ESCALATE, D.CROSS_CHECK})


class PolicyGuard(Protocol):
    @property
    def name(self) -> str: ...

    def apply(self, decision: Decision, f: FieldContext) -> Decision: ...


def _settle(decision: Decision, f: FieldContext, why: str, rule: str) -> Decision:
    """Stop spending: keep a verified value (flagged as unconfirmed), review anything else."""
    if f.state.status is S.VERIFIED:
        return replace(
            decision, kind=D.ACCEPT, reason=f"unconfirmed: {why}", rule=rule, target_tier=None
        )
    return replace(decision, kind=D.REVIEW, reason=why, rule=rule, target_tier=None)


class MissingCrossModelGuard:
    name = "G3"

    def apply(self, decision: Decision, f: FieldContext) -> Decision:
        if decision.kind is D.CROSS_CHECK and not f.has_cross_model:
            return _settle(decision, f, "no cross-check model configured", self.name)
        return decision


class LastRoundGuard:
    name = "G2"

    def apply(self, decision: Decision, f: FieldContext) -> Decision:
        if decision.kind in SPENDING and f.round + 1 >= f.profile.max_rounds:
            return _settle(decision, f, "max rounds", self.name)
        return decision


class BudgetGuardRule:
    name = "G1"

    def apply(self, decision: Decision, f: FieldContext) -> Decision:
        if decision.kind in SPENDING and not f.can_afford(decision):
            return _settle(decision, f, "budget", self.name)
        return decision


DEFAULT_GUARDS: tuple[PolicyGuard, ...] = (
    MissingCrossModelGuard(),
    LastRoundGuard(),
    BudgetGuardRule(),
)


class GuardedPolicy(EscalationPolicy):
    def __init__(
        self, inner: EscalationPolicy, guards: Sequence[PolicyGuard] = DEFAULT_GUARDS
    ) -> None:
        super().__init__(inner.rules)
        self.inner, self.guards = inner, tuple(guards)

    def decide(self, f: FieldContext) -> Decision:
        decision = self.inner.decide(f)
        for guard in self.guards:
            decision = guard.apply(decision, f)
        return decision


def build_policy(
    names: Sequence[str] | None = None, registry: StrategyRegistry[PolicyRule] = POLICY_RULES
) -> GuardedPolicy:
    """The policy from rule names (``settings.engine.policy_rules``), always with the guards."""
    rules = [registry.get(name) for name in (names or DEFAULT_POLICY_RULES)]
    return GuardedPolicy(EscalationPolicy(rules))
