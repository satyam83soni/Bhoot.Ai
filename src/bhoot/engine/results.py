"""Build the ``ExtractionResult`` developers get from the final graph state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from bhoot.domain.cost import CostReport
from bhoot.domain.decisions import DecisionKind
from bhoot.domain.document import Document
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.job import JobSpec
from bhoot.domain.result import (
    AttemptSummary,
    CheckSummary,
    CostSummary,
    DocumentInfo,
    ExtractionResult,
    FieldResult,
    FieldTrace,
    JobStatus,
)
from bhoot.engine.data import assemble


class ResultAssembler:
    def build(
        self, job: JobSpec, state: Mapping[str, Any], *, include_trace: bool
    ) -> ExtractionResult[Any]:
        states: Mapping[str, FieldState] = state.get("field_states", {})
        fields = {path: self._field(fs, include_trace) for path, fs in sorted(states.items())}
        values = {p: (f.value if f.status != "absent" else None) for p, f in fields.items()}
        doc_checks = state.get("doc_checks", ())
        reasons = [
            f"{p}: {f.review_reason}" for p, f in fields.items() if f.status == "needs_review"
        ]
        reasons += [c.message or c.check_id for c in doc_checks if c.blocks]
        verified_tiers = Counter(
            fs.current.tier
            for fs in states.values()
            if fs.final is DecisionKind.ACCEPT and fs.current
        )
        cost: CostReport = state.get("cost") or CostReport()
        summary = CostSummary.of(cost).model_copy(
            update={"fields_by_tier": {int(t): n for t, n in sorted(verified_tiers.items())}}
        )
        return ExtractionResult[Any](
            job_id=job.job_id,
            status=JobStatus(state.get("status", JobStatus.NEEDS_REVIEW.value)),
            data=assemble(values),
            fields=fields,
            checks=tuple(CheckSummary.of(c) for c in doc_checks),
            review_reasons=tuple(reasons),
            cost=summary,
            timings_ms={k: int(v) for k, v in state.get("timings_ms", {}).items()},
            versions=job.versions,
            document=_doc_info(state.get("document")),
            warnings=tuple(dict.fromkeys(state.get("warnings", ()))),
        )

    def failed(
        self, job: JobSpec, error: BaseException, state: Mapping[str, Any] | None = None
    ) -> ExtractionResult[Any]:
        state = state or {}
        cost: CostReport = state.get("cost") or CostReport()
        return ExtractionResult[Any](
            job_id=job.job_id,
            status=JobStatus.FAILED,
            cost=CostSummary.of(cost),
            versions=job.versions,
            document=_doc_info(state.get("document")),
            warnings=tuple(dict.fromkeys(state.get("warnings", ()))),
            error=f"{type(error).__name__}: {error}",
        )

    @staticmethod
    def _field(fs: FieldState, include_trace: bool) -> FieldResult:
        cand = fs.current
        evidence = cand.evidence if cand else None
        if fs.final is DecisionKind.ACCEPT:
            status = "verified"
        elif fs.status is FieldStatus.ABSENT:
            status = "absent"
        else:
            status = "needs_review"
        page = evidence.page if evidence else (cand.page if cand else None)
        source = (
            evidence.source.value
            if evidence
            else ("code" if cand and cand.is_code else ("llm" if cand else None))
        )
        review = fs.history[-1][2] if status == "needs_review" and fs.history else None
        trace = None
        if include_trace:
            trace = FieldTrace(
                attempts=tuple(
                    AttemptSummary(
                        round=a.round,
                        tier=int(a.tier),
                        kind=a.kind.value if a.kind else None,
                        model=a.model,
                        outcome=a.outcome.value,
                        reason=a.reason,
                        cost_usd=a.cost_usd,
                    )
                    for a in fs.attempts
                ),
                confidence_signals=dict(fs.confidence_signals),
                decisions=fs.decision_log,
                alternatives=tuple(
                    {"value": str(a.value), "model": a.model} for a in fs.alternatives
                ),
                match={"score": evidence.match_score, "grounded": evidence.grounded}
                if evidence
                else None,
                ocr_min_confidence=evidence.ocr_min_confidence if evidence else None,
            )
        return FieldResult(
            path=fs.path.raw,
            value=fs.value if status != "absent" else None,
            status=status,
            page=page + 1 if page is not None else None,
            quote=evidence.quote if evidence else (cand.quote if cand else None),
            bbox=evidence.bbox.as_tuple() if evidence and evidence.bbox else None,
            source=source,
            tier=int(cand.tier) if cand else None,
            confidence=fs.confidence,
            checks=tuple(CheckSummary.of(c) for c in fs.checks),
            attempts=len(fs.attempts),
            review_reason=review,
            trace=trace,
        )


def _doc_info(doc: Document | None) -> DocumentInfo | None:
    if doc is None:
        return None
    return DocumentInfo(
        sha256=doc.sha256,
        pages=doc.page_count,
        mime=doc.mime,
        page_kinds={p.number: p.kind.value for p in doc.pages},
    )
