"""Confidence from the evidence behind a value (LLD §9.15.1). Every signal is recorded, so
``explain()`` can show why a field scored what it did."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

from bhoot.domain.checks import CheckStatus
from bhoot.domain.evidence import EvidenceSource
from bhoot.domain.field_state import FieldState
from bhoot.domain.modes import Tier
from bhoot.domain.schema import Criticality

CODE_SOURCES = frozenset({EvidenceSource.CODE, EvidenceSource.OCR_CODE})


@dataclass(frozen=True, slots=True)
class ConfidenceWeights:
    exact_quote: float = 0.35
    value_in_quote: float = 0.25
    rules_pass: float = 0.30
    document_rule_bonus: float = 0.10
    low_ocr_penalty: float = 0.20
    confirmed_floor: float = 0.97  # crop re-read or a second model agreed
    scanned_cap: float = 0.85
    per_repair: float = 0.05


class ConfidenceScorer:
    def __init__(self, weights: ConfidenceWeights | None = None) -> None:
        self.w = weights or ConfidenceWeights()

    def score(
        self,
        fs: FieldState,
        *,
        fuzzy_threshold: float,
        critical_word_confidence: float,
        document_rule_ids: Iterable[str] = (),
        cross_agrees: bool | None = None,
    ) -> FieldState:
        cand = fs.current
        if cand is None:
            return fs
        w = self.w
        signals: dict[str, float] = {}
        evidence = cand.evidence
        blocking = any(c.blocks for c in fs.checks)
        if not blocking and (
            cand.tier is Tier.CODE or (evidence and evidence.source in CODE_SOURCES)
        ):
            signals["read_by_code"] = 1.0
        elif evidence is not None and evidence.grounded:
            if evidence.match_score >= 99:
                signals["exact_quote"] = w.exact_quote
            elif fuzzy_threshold < 100:
                share = (evidence.match_score - fuzzy_threshold) / (100 - fuzzy_threshold)
                signals["fuzzy_quote"] = round(w.exact_quote * max(0.0, min(1.0, share)), 4)
            signals["value_in_quote"] = w.value_in_quote
        elif evidence is not None:  # no text on the page: only the quote itself was checked
            signals["value_in_quote"] = w.value_in_quote
        if not blocking and fs.checks:
            signals["rules_pass"] = w.rules_pass
            doc_ids = set(document_rule_ids)
            if any(c.check_id in doc_ids and c.status is CheckStatus.PASS for c in fs.checks):
                signals["document_rule"] = w.document_rule_bonus
        if (
            evidence is not None
            and evidence.ocr_min_confidence is not None
            and evidence.ocr_min_confidence < critical_word_confidence
            and fs.spec.criticality is Criticality.CRITICAL
        ):
            signals["low_ocr_confidence"] = -w.low_ocr_penalty
        repairs = max(0, len(fs.attempts) - 1)
        if repairs:
            signals["repairs"] = -w.per_repair * repairs

        confidence = sum(signals.values())
        if evidence is not None and not evidence.grounded and "read_by_code" not in signals:
            confidence = min(confidence, w.scanned_cap)
            signals["scanned_cap"] = w.scanned_cap
        if fs.crop_checked and not blocking:
            confidence = max(confidence, w.confirmed_floor)
            signals["crop_confirmed"] = w.confirmed_floor
        if cross_agrees:
            confidence = max(confidence, w.confirmed_floor)
            signals["cross_model_agrees"] = w.confirmed_floor
        confidence = round(max(0.0, min(1.0, confidence)), 4)
        return fs.with_confidence(confidence, MappingProxyType(signals))
