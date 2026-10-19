from __future__ import annotations

import itertools

import pytest

from bhoot.domain.candidate import CandidateRole, CandidateValue
from bhoot.domain.checks import CheckResult, CheckStatus
from bhoot.domain.decisions import DecisionKind
from bhoot.domain.errors import IllegalTransitionError
from bhoot.domain.evidence import Evidence, EvidenceSource
from bhoot.domain.field_state import ALLOWED_TRANSITIONS, FieldState, FieldStatus
from bhoot.domain.modes import Tier
from bhoot.domain.schema import Criticality, FieldPath, FieldSpec
from bhoot.domain.work import WorkKind

S = FieldStatus
PATH = FieldPath("closing_balance")
SPEC = FieldSpec(PATH, "number", "Decimal", Criticality.CRITICAL)


def _state(status: FieldStatus = S.PENDING) -> FieldState:
    return FieldState(PATH, SPEC, status=status)


def _cand(
    raw: str = "48,250.75", kind: WorkKind | None = WorkKind.EXTRACT, role=CandidateRole.PRIMARY
) -> CandidateValue:
    return CandidateValue(
        PATH, raw, "r0-summary-3-extract-0", 0, Tier.CHEAP, model="m", role=role, source_kind=kind
    )


@pytest.mark.parametrize(("src", "dst"), list(itertools.product(FieldStatus, FieldStatus)))
def test_transition_table_is_enforced(src: FieldStatus, dst: FieldStatus) -> None:
    state = _state(src)
    if dst in ALLOWED_TRANSITIONS[src]:
        assert state.transition(dst, "test").status is dst
    else:
        with pytest.raises(IllegalTransitionError, match="not allowed"):
            state.transition(dst, "test")


def test_verified_is_only_reachable_from_validated() -> None:
    sources = [s for s, targets in ALLOWED_TRANSITIONS.items() if S.VERIFIED in targets]
    assert sources == [S.VALIDATED]


def test_needs_review_is_a_dead_end() -> None:
    assert ALLOWED_TRANSITIONS[S.NEEDS_REVIEW] == frozenset()


def test_happy_path_and_history() -> None:
    state = _state().with_candidate(_cand())
    state = state.transition(S.VALIDATED, "rules pass").transition(S.VERIFIED, "quote found")
    state = state.decide(DecisionKind.ACCEPT, "verified", "rule5", 0)
    assert state.status is S.VERIFIED
    assert state.final is DecisionKind.ACCEPT
    assert state.is_terminal
    assert [h[1] for h in state.history] == [S.CANDIDATE, S.VALIDATED, S.VERIFIED]
    assert state.decision_log == ("r0:rule5:accept",)
    assert state.attempts[-1].outcome is S.VERIFIED


def test_state_is_immutable() -> None:
    original = _state()
    changed = original.with_candidate(_cand())
    assert original.status is S.PENDING
    assert original.current is None
    assert changed.current is not None
    with pytest.raises(AttributeError):
        original.status = S.VERIFIED  # type: ignore[misc]


def test_new_candidate_reopens_and_clears_checks() -> None:
    check = CheckResult("x.rule", CheckStatus.FAIL, paths=(PATH,))
    state = (
        _state().with_candidate(_cand()).transition(S.FAILED_CHECKS, "rule failed", checks=(check,))
    )
    assert state.attempts[-1].outcome is S.FAILED_CHECKS
    repaired = state.with_candidate(_cand("48,250.76", WorkKind.REPAIR))
    assert repaired.status is S.CANDIDATE
    assert repaired.checks == ()
    assert len(repaired.attempts) == 2
    assert repaired.attempts[-1].kind is WorkKind.REPAIR


def test_later_rule_failure_reopens_an_accepted_field() -> None:
    accepted = (
        _state().with_candidate(_cand()).transition(S.VALIDATED).transition(S.VERIFIED)
        .decide(DecisionKind.ACCEPT, "ok", "rule5", 0)
    )  # fmt: skip
    reopened = accepted.transition(S.FAILED_CHECKS, "running balance broke")
    assert reopened.final is None
    assert not reopened.is_terminal


def test_only_verified_fields_can_be_accepted() -> None:
    with pytest.raises(IllegalTransitionError, match="only VERIFIED"):
        _state().with_candidate(_cand()).decide(DecisionKind.ACCEPT, "x", "rule5", 0)


def test_review_and_absent_decisions_close_the_field() -> None:
    review = _state().with_candidate(_cand()).decide(DecisionKind.REVIEW, "budget", "G1", 2)
    assert review.status is S.NEEDS_REVIEW
    assert review.final is DecisionKind.REVIEW
    absent = _state().decide(DecisionKind.ABSENT, "not on page", "rule7", 1)
    assert absent.status is S.ABSENT
    assert absent.is_terminal
    found_later = absent.with_candidate(_cand())
    assert found_later.status is S.CANDIDATE
    assert not found_later.is_terminal


def test_non_final_decisions_only_log() -> None:
    state = _state().with_candidate(_cand()).transition(S.FAILED_CHECKS)
    after = state.decide(DecisionKind.REPAIR, "retry", "rule8", 1)
    assert after.status is S.FAILED_CHECKS
    assert after.final is None
    assert after.decision_log == ("r1:rule8:repair",)


def test_alternatives_do_not_change_status() -> None:
    state = _state().with_candidate(_cand()).transition(S.VALIDATED).transition(S.VERIFIED)
    alt = _cand(kind=WorkKind.CROSSCHECK, role=CandidateRole.ALTERNATIVE)
    with_alt = state.with_alternative(alt)
    assert with_alt.status is S.VERIFIED
    assert with_alt.alternatives == (alt,)
    with pytest.raises(ValueError, match="with_alternative"):
        state.with_candidate(alt)


def test_crop_reread_is_remembered() -> None:
    state = _state().with_candidate(_cand()).with_candidate(_cand(kind=WorkKind.CROP))
    assert state.crop_checked


def test_evidence_grounding() -> None:
    assert Evidence(0, "q", EvidenceSource.CODE).grounded
    assert not Evidence(0, "q", EvidenceSource.LLM).grounded
    assert Evidence(0, "q", EvidenceSource.LLM, char_span=(3, 9)).grounded


def test_candidate_helpers() -> None:
    c = _cand()
    assert c.typed("x").value == "x"
    failed = c.failed_coercion("not a number")
    assert failed.value is None
    assert failed.coercion_error == "not a number"
    assert not c.is_code
    assert DecisionKind.REVIEW.is_final
    assert not DecisionKind.REPAIR.is_final
