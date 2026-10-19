"""Policy: one row per escalation rule and guard, confidence signals, and a full decide pass."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest

from bhoot.config.modes import MODE_PROFILES
from bhoot.domain.candidate import CandidateRole, CandidateValue
from bhoot.domain.checks import CheckResult, CheckStatus
from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import Document, Page, PageKind
from bhoot.domain.evidence import Evidence, EvidenceSource
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.modes import Mode, Tier
from bhoot.domain.schema import Criticality, FieldPath, FieldSpec
from bhoot.domain.work import WorkKind
from bhoot.engine.policy import (
    ConfidenceScorer,
    FieldContext,
    build_policy,
    decide_fields,
)

BALANCED = MODE_PROFILES[Mode.BALANCED]
ECONOMY = MODE_PROFILES[Mode.ECONOMY]
S, D = FieldStatus, DecisionKind
CRIT, IMP, NORM = Criticality.CRITICAL, Criticality.IMPORTANT, Criticality.NORMAL


def _cand(value: Any = Decimal("10"), *, tier: Tier = Tier.CHEAP, evidence: Evidence | None = None,
          role: CandidateRole = CandidateRole.PRIMARY, kind: WorkKind = WorkKind.EXTRACT) -> CandidateValue:  # fmt: skip
    c = CandidateValue(FieldPath("f"), str(value), "r0-x", 0, tier, value=value, quote="q", page=0,
                       model="m", role=role, source_kind=kind)  # fmt: skip
    return c.with_evidence(evidence) if evidence else c


def _state(
    status: FieldStatus, crit: Criticality = NORM, *, required: bool = False, **kw: Any
) -> FieldState:
    spec = FieldSpec(FieldPath("f"), "number", "Decimal", crit, required=required)
    fs = FieldState(FieldPath("f"), spec)
    if status is S.PENDING:
        return replace(fs, **kw)
    fs = fs.with_candidate(
        kw.pop("candidate", _cand(evidence=Evidence(0, "q", EvidenceSource.LLM, (0, 1))))
    )
    if status is S.CANDIDATE:
        return replace(fs, **kw)
    if status in (S.FAILED_CHECKS,):
        return replace(fs.transition(S.FAILED_CHECKS, "bad"), **kw)
    fs = fs.transition(S.VALIDATED)
    if status is S.UNGROUNDED:
        return replace(fs.transition(S.UNGROUNDED, "no quote"), **kw)
    if status is S.VERIFIED:
        return replace(fs.transition(S.VERIFIED), **kw)
    return replace(fs, **kw)


def _log(n: int, kind: str = "repair") -> tuple[str, ...]:
    return tuple(f"r{i}:rule8:{kind}" for i in range(n))


LOW_OCR = Evidence(0, "q", EvidenceSource.LLM, (0, 1), ocr_min_confidence=0.6)
POLICY = build_policy()

CASES = [
    # id, state, context overrides, expected kind, expected rule
    ("r1 disagree", _state(S.VERIFIED, CRIT, cross_checked=True), {"alternatives_agree": False}, D.REVIEW, "rule1"),
    ("r2 agree", _state(S.VERIFIED, CRIT, cross_checked=True), {"alternatives_agree": True}, D.ACCEPT, "rule2"),
    ("r3 low ocr", _state(S.VERIFIED, CRIT, candidate=_cand(evidence=LOW_OCR)), {"page_kind": PageKind.OCR_TEXT}, D.REPAIR, "rule3"),
    ("r3 not after crop", _state(S.VERIFIED, CRIT, candidate=_cand(evidence=LOW_OCR), crop_checked=True, confidence=0.99),
     {"page_kind": PageKind.OCR_TEXT}, D.ACCEPT, "rule5"),
    ("r4 cross-check", _state(S.VERIFIED, CRIT, confidence=0.5), {}, D.CROSS_CHECK, "rule4"),
    ("r4 not when confident", _state(S.VERIFIED, CRIT, confidence=0.9), {}, D.ACCEPT, "rule5"),
    ("r4 not below cross tier", _state(S.VERIFIED, IMP, confidence=0.1), {}, D.ACCEPT, "rule5"),
    ("r5 accept", _state(S.VERIFIED), {}, D.ACCEPT, "rule5"),
    ("r6 missing required", _state(S.PENDING, required=True), {"returned_null": True}, D.REPAIR, "rule6"),
    ("r6 errored item", _state(S.PENDING), {}, D.REPAIR, "rule6"),
    ("r7 absent", _state(S.PENDING), {"returned_null": True}, D.ABSENT, "rule7"),
    ("r8 repair", _state(S.FAILED_CHECKS), {}, D.REPAIR, "rule8"),
    ("r8 ungrounded", _state(S.UNGROUNDED, IMP, decision_log=_log(1)), {}, D.REPAIR, "rule8"),
    ("r9 escalate", _state(S.FAILED_CHECKS, IMP, decision_log=_log(2)), {}, D.ESCALATE, "rule9"),
    ("r9 pending exhausted", _state(S.PENDING, CRIT, required=True, decision_log=_log(2)), {}, D.ESCALATE, "rule9"),
    ("r10 strong tried", _state(S.FAILED_CHECKS, IMP, decision_log=(*_log(2), "r2:rule9:escalate")), {}, D.REVIEW, "rule10"),
    ("r10 mode caps tier", _state(S.UNGROUNDED, NORM, decision_log=_log(2)), {}, D.REVIEW, "rule10"),
    ("fallback", _state(S.CANDIDATE), {}, D.REVIEW, "fallback"),
    # guards
    ("G1 budget verified", _state(S.VERIFIED, CRIT, confidence=0.5), {"can_afford": lambda d: False}, D.ACCEPT, "G1"),
    ("G1 budget broken", _state(S.FAILED_CHECKS), {"can_afford": lambda d: False}, D.REVIEW, "G1"),
    ("G2 last round", _state(S.UNGROUNDED), {"round": 3}, D.REVIEW, "G2"),
    ("G2 keeps verified", _state(S.VERIFIED, CRIT, confidence=0.5), {"round": 3}, D.ACCEPT, "G2"),
    ("G3 no cross model, but VERIFIED", _state(S.VERIFIED, CRIT, confidence=0.5), {"has_cross_model": False}, D.ACCEPT, "rule5"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("state", "overrides", "kind", "rule"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_policy_table(
    state: FieldState, overrides: dict[str, Any], kind: DecisionKind, rule: str
) -> None:
    ctx = FieldContext(state=state, profile=BALANCED, round=overrides.pop("round", 0), **overrides)
    decision = POLICY.decide(ctx)
    assert (decision.kind, decision.rule) == (kind, rule), decision.reason


def test_guard_g3_and_mode_limits() -> None:
    from bhoot.engine.policy import MissingCrossModelGuard

    ctx = FieldContext(
        state=_state(S.VERIFIED, CRIT, confidence=0.5),
        profile=BALANCED,
        round=0,
        has_cross_model=False,
    )
    asked = Decision(FieldPath("f"), D.CROSS_CHECK, "low", "custom", Tier.CROSS)
    settled = MissingCrossModelGuard().apply(asked, ctx)
    assert (settled.kind, settled.rule) == (D.ACCEPT, "G3")
    assert settled.reason.startswith("unconfirmed")
    economy = FieldContext(
        state=_state(S.FAILED_CHECKS, CRIT, decision_log=_log(1)), profile=ECONOMY, round=0
    )
    assert POLICY.decide(economy).kind is D.REVIEW  # economy never goes past one cheap repair


def test_custom_rule_order_from_settings() -> None:
    only_accept = build_policy(["rule5", "rule10"])
    ctx = FieldContext(state=_state(S.VERIFIED, CRIT, confidence=0.1), profile=BALANCED, round=0)
    assert only_accept.decide(ctx).rule == "rule5"  # rule4 left out: no cross-check


# ---------------------------------------------------------------- confidence
def _scored(fs: FieldState, **kw: Any) -> FieldState:
    return ConfidenceScorer().score(fs, fuzzy_threshold=90, critical_word_confidence=0.9, **kw)


PASS = CheckResult("doc.balance", CheckStatus.PASS, paths=(FieldPath("f"),))


def test_confidence_signals() -> None:
    exact = _scored(_state(S.VERIFIED, candidate=_cand(evidence=Evidence(0, "q", EvidenceSource.LLM, (0, 1))), checks=(PASS,)),
                    document_rule_ids=["doc.balance"])  # fmt: skip
    assert exact.confidence == 1.0  # 0.35 + 0.25 + 0.30 + 0.10
    assert set(exact.confidence_signals) == {
        "exact_quote",
        "value_in_quote",
        "rules_pass",
        "document_rule",
    }

    fuzzy = _scored(
        _state(
            S.VERIFIED,
            candidate=_cand(evidence=Evidence(0, "q", EvidenceSource.LLM, (0, 1), match_score=95)),
        )
    )
    assert fuzzy.confidence_signals["fuzzy_quote"] == pytest.approx(0.175)

    code = _scored(_state(S.VERIFIED, candidate=_cand(tier=Tier.CODE)))
    assert code.confidence == 1.0

    scanned = _scored(
        _state(
            S.VERIFIED,
            candidate=_cand(evidence=Evidence(0, "q", EvidenceSource.LLM, match_score=0)),
            checks=(PASS,),
        )
    )
    assert scanned.confidence == pytest.approx(0.55)  # value in quote + rules, no grounding
    capped = _scored(replace(scanned, checks=(PASS,)), document_rule_ids=["doc.balance"])
    assert capped.confidence <= 0.85

    low_ocr = _scored(_state(S.VERIFIED, CRIT, candidate=_cand(evidence=LOW_OCR)))
    assert low_ocr.confidence_signals["low_ocr_confidence"] == -0.2

    agreed = _scored(_state(S.VERIFIED, candidate=_cand(evidence=LOW_OCR)), cross_agrees=True)
    assert agreed.confidence == 0.97


# ---------------------------------------------------------------- decide pass
def _doc() -> Document:
    return Document("d", "s", "application/pdf", (Page(0, 595, 842, PageKind.TEXT, "text"),))


def test_decide_pass_compares_alternatives_and_closes_fields() -> None:
    crit = _state(S.VERIFIED, CRIT, confidence=0.5, cross_checked=True)
    alt_same = _cand(Decimal("10.001"), role=CandidateRole.ALTERNATIVE, kind=WorkKind.CROSSCHECK)
    alt_diff = _cand(Decimal("99"), role=CandidateRole.ALTERNATIVE, kind=WorkKind.CROSSCHECK)
    states = {
        "agree": crit.with_alternative(alt_same),
        "disagree": crit.with_alternative(alt_diff),
        "absent": _state(S.PENDING),
        "broken": _state(S.FAILED_CHECKS),
        "done": replace(_state(S.VERIFIED), final=D.ACCEPT),
        "budget": _state(S.VERIFIED, CRIT, confidence=0.1),
    }
    result = decide_fields(
        states, doc=_doc(), profile=BALANCED, round_=0, policy=POLICY,
        returned_null={"absent"}, can_afford=lambda d: d.kind is not D.CROSS_CHECK,
    )  # fmt: skip
    out = result.field_states
    assert out["agree"].final is D.ACCEPT
    assert out["agree"].cross_checked
    assert out["disagree"].status is S.NEEDS_REVIEW
    assert out["absent"].status is S.ABSENT
    assert out["broken"].final is None  # repair is not final
    assert out["broken"].decision_log[-1] == "r0:rule8:repair"
    assert out["done"] is states["done"]  # terminal fields are left alone
    assert out["budget"].final is D.ACCEPT
    assert result.warnings == ("budget: accepted unconfirmed: budget",)
    assert [d.kind for d in result.decisions] == [D.ACCEPT, D.REVIEW, D.ABSENT, D.REPAIR, D.ACCEPT]
