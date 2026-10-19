"""One decision round over every open field: compare second answers, score, decide, apply."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace

from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import Document, PageKind
from bhoot.domain.field_state import FieldState
from bhoot.domain.modes import ModeProfile
from bhoot.engine.policy.confidence import ConfidenceScorer
from bhoot.engine.policy.rules import EscalationPolicy, FieldContext, _always
from bhoot.engine.verification.matching import ValueComparator


@dataclass(frozen=True)
class DecideResult:
    field_states: dict[str, FieldState]
    decisions: tuple[Decision, ...]
    warnings: tuple[str, ...]


def page_kind_of(fs: FieldState, doc: Document) -> PageKind | None:
    cand = fs.current
    page = cand.evidence.page if cand and cand.evidence else (cand.page if cand else None)
    if page is None or not 0 <= page < doc.page_count:
        return None
    return doc.page(page).kind


def decide_fields(
    field_states: Mapping[str, FieldState],
    *,
    doc: Document,
    profile: ModeProfile,
    round_: int,
    policy: EscalationPolicy,
    returned_null: Collection[str] = (),
    has_cross_model: bool = True,
    can_afford: Callable[[Decision], bool] = _always,
    critical_word_confidence: float = 0.90,
    document_rule_ids: Iterable[str] = (),
    scorer: ConfidenceScorer | None = None,
    comparator: ValueComparator | None = None,
) -> DecideResult:
    scorer = scorer or ConfidenceScorer()
    comparator = comparator or ValueComparator()
    doc_rules = tuple(document_rule_ids)
    out = dict(field_states)
    decisions: list[Decision] = []
    warnings: list[str] = []
    for path, fs in field_states.items():
        if fs.is_terminal:
            continue
        agree: bool | None = None
        if fs.alternatives and fs.current is not None:
            alt = fs.alternatives[-1]
            agree = comparator.equal(fs.current.value, alt.value, fs.spec)
            fs = replace(fs, cross_checked=True)
        fs = scorer.score(
            fs,
            fuzzy_threshold=profile.fuzzy_quote_threshold,
            critical_word_confidence=critical_word_confidence,
            document_rule_ids=doc_rules,
            cross_agrees=agree,
        )
        ctx = FieldContext(
            state=fs,
            profile=profile,
            round=round_,
            page_kind=page_kind_of(fs, doc),
            returned_null=path in returned_null,
            has_cross_model=has_cross_model,
            alternatives_agree=agree,
            critical_word_confidence=critical_word_confidence,
            can_afford=can_afford,
        )
        decision = policy.decide(ctx)
        out[path] = fs.decide(decision.kind, decision.reason, decision.rule, round_)
        decisions.append(decision)
        if decision.kind is DecisionKind.ACCEPT and decision.reason.startswith("unconfirmed"):
            warnings.append(f"{path}: accepted {decision.reason}")
    return DecideResult(out, tuple(decisions), tuple(warnings))
