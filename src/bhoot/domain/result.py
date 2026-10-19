"""What developers get back: ``ExtractionResult`` (versioned JSON contract ``result_version "1"``).

Changes to this module must be additive: new optional fields only, never renames or removals.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from bhoot.domain.checks import CheckResult
from bhoot.domain.cost import CostReport
from bhoot.domain.versions import ResolvedVersions

T = TypeVar("T", bound=BaseModel)

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class JobStatus(StrEnum):
    SUCCEEDED = "succeeded"  # every required field verified or absent
    NEEDS_REVIEW = "needs_review"  # at least one field needs a human
    FAILED = "failed"  # bad file, provider down, timeout


class CheckSummary(BaseModel):
    model_config = _FROZEN

    check_id: str
    status: Literal["pass", "fail", "skip", "warn"]
    severity: Literal["error", "warning"] = "error"
    paths: tuple[str, ...] = ()
    message: str = ""

    @classmethod
    def of(cls, check: CheckResult) -> CheckSummary:
        return cls(
            check_id=check.check_id,
            status=check.status.value,
            severity=check.severity.value,
            paths=tuple(p.raw for p in check.paths),
            message=check.message,
        )


class AttemptSummary(BaseModel):
    model_config = _FROZEN

    round: int
    tier: int
    kind: str | None  # extract | repair | crop | crosscheck; None for Tier 0
    model: str | None
    outcome: str
    reason: str | None = None
    cost_usd: Decimal = Decimal("0")


class FieldTrace(BaseModel):
    """Everything needed to explain one field. Only present when include_trace=True."""

    model_config = _FROZEN

    attempts: tuple[AttemptSummary, ...] = ()
    confidence_signals: dict[str, float] = Field(default_factory=dict)
    decisions: tuple[str, ...] = ()  # e.g. "r1:rule8:repair"
    alternatives: tuple[dict[str, Any], ...] = ()
    match: dict[str, Any] | None = None
    ocr_min_confidence: float | None = None


class FieldResult(BaseModel):
    model_config = _FROZEN

    path: str
    value: Any = None
    status: Literal["verified", "needs_review", "absent"]
    page: int | None = None  # 1-based
    quote: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    source: Literal["code", "ocr_code", "llm", "crop"] | None = None
    tier: int | None = None
    confidence: float = 0.0
    checks: tuple[CheckSummary, ...] = ()
    attempts: int = 0
    review_reason: str | None = None
    trace: FieldTrace | None = None


class CostSummary(BaseModel):
    model_config = _FROZEN

    total_usd: Decimal = Decimal("0")
    llm_calls: int = 0
    cache_hits: int = 0
    by_tier: dict[int, Decimal] = Field(default_factory=dict)
    fields_by_tier: dict[int, int] = Field(default_factory=dict)

    @classmethod
    def of(cls, report: CostReport) -> CostSummary:
        return cls(
            total_usd=report.total_usd,
            llm_calls=report.llm_calls,
            cache_hits=report.cache_hits,
            by_tier={int(t): v for t, v in report.by_tier.items()},
            fields_by_tier={int(t): n for t, n in report.fields_by_tier.items()},
        )


class DocumentInfo(BaseModel):
    model_config = _FROZEN

    sha256: str
    pages: int
    mime: str
    page_kinds: dict[int, str] = Field(default_factory=dict)  # 1-based page -> kind


@dataclass(frozen=True)
class FieldExplanation:
    path: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def __str__(self) -> str:
        return self.text


class ExtractionResult(BaseModel, Generic[T]):
    model_config = _FROZEN

    result_version: Literal["1"] = "1"
    job_id: str
    status: JobStatus
    data: dict[str, Any] = Field(default_factory=dict)
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    checks: tuple[CheckSummary, ...] = ()
    review_reasons: tuple[str, ...] = ()
    cost: CostSummary = Field(default_factory=CostSummary)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    versions: ResolvedVersions
    document: DocumentInfo | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None

    _model: type[T] | None = PrivateAttr(default=None)

    # ---------------------------------------------------------------- typed access
    def bind(self, model: type[T]) -> ExtractionResult[T]:
        """Attach the generated model class so ``typed`` returns a typed object."""
        bound: ExtractionResult[T] = self.model_copy()
        bound._model = model
        return bound

    @property
    def typed(self) -> T:
        if self._model is None:
            raise AttributeError(
                "no model bound: pass a generated SchemaRef or call to_model(Model)"
            )
        return self._model.model_validate(self.data)

    def to_model(self, model: type[T]) -> T:
        return model.model_validate(self.data)

    # ---------------------------------------------------------------- helpers
    @property
    def succeeded(self) -> bool:
        return self.status is JobStatus.SUCCEEDED

    def needs_review_fields(self) -> list[FieldResult]:
        return [f for f in self.fields.values() if f.status == "needs_review"]

    def without_traces(self) -> ExtractionResult[T]:
        fields = {k: f.model_copy(update={"trace": None}) for k, f in self.fields.items()}
        out: ExtractionResult[T] = self.model_copy(update={"fields": fields})
        out._model = self._model
        return out

    def to_json(self, *, include_fields: bool = True, indent: int | None = 2) -> str:
        exclude = None if include_fields else {"fields"}
        return self.model_dump_json(indent=indent, exclude=exclude, by_alias=True)

    def explain(self, path: str) -> FieldExplanation:
        field_result = self.fields.get(path)
        if field_result is None:
            close = [p for p in self.fields if path in p][:3]
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise KeyError(f"no field {path!r} in this result{hint}")
        return explain_field(field_result)


def explain_field(f: FieldResult) -> FieldExplanation:
    lines = [
        f"{f.path} = {f.value!s:<20} status: {f.status.upper()}   confidence: {f.confidence:.2f}"
    ]
    if f.quote is not None or f.page is not None:
        where = f"page {f.page}" if f.page else "page ?"
        box = f" · bbox {tuple(round(v) for v in f.bbox)}" if f.bbox else ""
        lines.append(
            f"  evidence   {where} · {f.quote!r}{box} · source: {f.source} (tier {f.tier})"
        )
    for c in f.checks:
        mark = {"pass": "ok", "fail": "FAILED", "warn": "warning", "skip": "skipped"}[c.status]
        lines.append(
            f"  check      {mark}: {c.check_id}" + (f" ({c.message})" if c.message else "")
        )
    if f.review_reason:
        lines.append(f"  review     {f.review_reason}")
    t = f.trace
    if t is None:
        lines.append(f"  attempts   {f.attempts} (run with include_trace=True for details)")
        return FieldExplanation(f.path, tuple(lines))
    if t.match:
        lines.append("  match      " + ", ".join(f"{k}={v}" for k, v in t.match.items()))
    for a in t.attempts:
        lines.append(
            f"  attempt    r{a.round} {a.kind or 'code'} tier{a.tier} {a.model or ''} -> {a.outcome}"
            + (f" ({a.reason})" if a.reason else "")
            + (f" ${a.cost_usd}" if a.cost_usd else "")
        )
    for d in t.decisions:
        lines.append(f"  decision   {d}")
    for alt in t.alternatives:
        lines.append(f"  cross      {alt}")
    if t.confidence_signals:
        signals = " · ".join(f"{k} {v:+.2f}" for k, v in t.confidence_signals.items())
        lines.append(f"  confidence {signals} -> {f.confidence:.2f}")
    return FieldExplanation(f.path, tuple(lines))
