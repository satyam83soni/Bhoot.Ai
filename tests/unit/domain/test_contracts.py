"""Job, result, pack, rule and option contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from bhoot.domain.checks import CheckResult, CheckStatus, Severity
from bhoot.domain.cost import Budget, CostEntry, CostReport, Usage
from bhoot.domain.document import PageKind
from bhoot.domain.example import Example
from bhoot.domain.hashing import canonical_json, content_hash
from bhoot.domain.job import JobSpec
from bhoot.domain.locales import Locale
from bhoot.domain.modes import Mode, ModeProfile, Tier
from bhoot.domain.options import ExtractOptions
from bhoot.domain.pack import DocPack, SectionSpec
from bhoot.domain.prompt import make_prompt_template
from bhoot.domain.result import (
    AttemptSummary,
    CheckSummary,
    CostSummary,
    ExtractionResult,
    FieldResult,
    FieldTrace,
    JobStatus,
)
from bhoot.domain.rules import RuleContext, RuleScope, fail, pass_, rule, warn
from bhoot.domain.schema import Criticality, SchemaDefinition
from bhoot.domain.versions import ResolvedVersions

VERSIONS = ResolvedVersions(engine="0.1.0", schema="invoice@1", schema_hash="sha256:x")


# ---------------------------------------------------------------- hashing, options, prompts
def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": Decimal("1.50")}) == '{"a":"1.50","b":1}'
    assert content_hash({"x": (1, 2)}) == content_hash({"x": [1, 2]})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_extract_options_validation_and_cache_subset() -> None:
    opts = ExtractOptions(pages=(1, 2), password="secret")
    assert "secret" not in repr(opts)
    assert set(opts.cache_relevant()) == {
        "pages",
        "force_vision_pages",
        "max_pages",
        "pack_options",
    }
    with pytest.raises(ValidationError):
        ExtractOptions(pages=(0,))
    with pytest.raises(ValidationError):
        ExtractOptions(unknown=True)  # type: ignore[call-arg]


def test_prompt_hash_ignores_name_but_not_body() -> None:
    a = make_prompt_template(
        name="a", version=1, body="Extract {{ section }}", variables=["section"]
    )
    b = make_prompt_template(
        name="b", version=7, body="Extract {{ section }}", variables=["section"]
    )
    c = make_prompt_template(name="a", version=1, body="Extract all", variables=["section"])
    assert a.content_hash == b.content_hash != c.content_hash
    assert a.ref == "a@1"


def test_example_needs_exactly_one_input() -> None:
    Example("e1", PageKind.TEXT, {"x": 1}, input_text="hello")
    with pytest.raises(ValueError, match="exactly one"):
        Example("e2", PageKind.TEXT, {"x": 1})


# ---------------------------------------------------------------- rules
@rule("claim.amount_positive", scope=RuleScope.FIELD, applies_to=("amount",))
def amount_positive(ctx: RuleContext) -> list[CheckResult]:
    return [pass_("amount")] if ctx.values["amount"] > 0 else [fail("amount", "must be positive")]


@rule(
    "claim.note", scope=RuleScope.DOCUMENT, applies_to=("items[*].cost",), severity=Severity.ERROR
)
def note(ctx: RuleContext) -> list[CheckResult]:
    return [warn("items[0].cost", "first item looks high", limit=100)]


def _ctx(**values: object) -> RuleContext:
    return RuleContext(
        values=values, data={"items": [{"cost": 5}, {"cost": 7}, "junk"]}, locale=Locale()
    )


def test_function_rules_fill_in_id_and_severity() -> None:
    [ok] = amount_positive.check(_ctx(amount=10))
    [bad] = amount_positive.check(_ctx(amount=-1))
    assert ok.status is CheckStatus.PASS
    assert bad.check_id == "claim.amount_positive"
    assert bad.blocks
    assert bad.message == "must be positive"
    [w] = note.check(_ctx())
    assert w.severity is Severity.WARNING  # warn() keeps its severity
    assert not w.blocks
    assert w.details["limit"] == 100


def test_rule_context_rows_and_rule_metadata() -> None:
    assert _ctx().rows("items") == [{"cost": 5}, {"cost": 7}]
    assert _ctx().rows("missing") == []
    assert amount_positive.scope is RuleScope.FIELD
    assert amount_positive.source_hash() == amount_positive.source_hash()
    assert "claim.amount_positive" in repr(amount_positive)


def test_rule_decorator_validates_inputs() -> None:
    with pytest.raises(ValueError, match="rule id"):
        rule("bad id", scope=RuleScope.FIELD, applies_to=("x",))
    with pytest.raises(ValueError, match="path"):
        rule("ok.id", scope=RuleScope.FIELD, applies_to=("a..b",))


# ---------------------------------------------------------------- packs & jobs
def _schema() -> SchemaDefinition:
    return SchemaDefinition("invoice", 1, {"type": "object"}, {}, "sha256:x")


def _pack(**kw: object) -> DocPack:
    args: dict[str, object] = {
        "name": "general.invoice",
        "version": 1,
        "schema": _schema(),
        "prompt": make_prompt_template(name="invoice", version=1, body="x", variables=[]),
        "sections": (SectionSpec(name="all", paths=("*",)),),
        "rules": (amount_positive,),
    }
    args.update(kw)
    return DocPack(**args)  # type: ignore[arg-type]


def test_pack_validation() -> None:
    pack = _pack()
    assert pack.ref == "general.invoice@1"
    assert pack.section("all").merge == "scalar_prefer_verified"
    with pytest.raises(KeyError):
        pack.section("nope")
    with pytest.raises(ValueError, match="at least one section"):
        _pack(sections=())
    with pytest.raises(ValueError, match="unique"):
        _pack(sections=(SectionSpec(name="a", paths=("*",)), SectionSpec(name="a", paths=("x",))))
    with pytest.raises(ValueError, match="rule ids"):
        _pack(rules=(amount_positive, amount_positive))


def test_job_never_keeps_the_password() -> None:
    profile = ModeProfile(
        mode=Mode.BALANCED, max_tier_by_criticality=dict.fromkeys(Criticality, Tier.STRONG),
        max_repairs_per_field=2, max_rounds=4, fuzzy_quote_threshold=90,
    )  # fmt: skip
    job = JobSpec(
        job_id="job_1", pack=_pack(), versions=VERSIONS, mode=profile, budget=Budget(),
        options=ExtractOptions(password="pw"), input_ref="blob", created_at=datetime.now(UTC),
        cache_key="k",
    )  # fmt: skip
    safe = job.without_secrets()
    assert safe.options.password is None
    assert job.options.password == "pw"
    assert safe.without_secrets() is safe


# ---------------------------------------------------------------- results
class Invoice(BaseModel):
    total: Decimal
    vendor: str


def _result(**kw: object) -> ExtractionResult[Invoice]:
    trace = FieldTrace(
        attempts=(AttemptSummary(round=0, tier=1, kind="extract", model="m", outcome="verified"),),
        confidence_signals={"exact_quote": 0.35, "rules_pass": 0.30},
        decisions=("r0:rule5:accept",),
        match={"method": "exact", "score": 100},
    )
    fields = {
        "total": FieldResult(
            path="total", value=Decimal("1250.00"), status="verified", page=1,
            quote="Total 1,250.00", bbox=(10.4, 20.0, 80.6, 30.0), source="llm", tier=1,
            confidence=0.98, checks=(CheckSummary(check_id="x", status="pass"),), attempts=1,
            trace=trace,
        ),
        "vendor": FieldResult(path="vendor", value=None, status="needs_review", review_reason="budget"),
    }  # fmt: skip
    args: dict[str, object] = {
        "job_id": "job_1", "status": JobStatus.NEEDS_REVIEW,
        "data": {"total": Decimal("1250.00"), "vendor": "Acme"}, "fields": fields, "versions": VERSIONS,
    }  # fmt: skip
    args.update(kw)
    return ExtractionResult[Invoice](**args)  # type: ignore[arg-type]


def test_result_json_round_trips() -> None:
    result = _result()
    payload = json.loads(result.to_json())
    assert payload["result_version"] == "1"
    assert payload["versions"]["schema"] == "invoice@1"
    assert payload["fields"]["total"]["value"] == "1250.00"
    again = ExtractionResult[Invoice].model_validate_json(result.to_json())
    assert again.fields["total"].quote == "Total 1,250.00"
    assert "fields" not in json.loads(result.to_json(include_fields=False))


def test_typed_access_needs_a_bound_model() -> None:
    result = _result()
    with pytest.raises(AttributeError, match="no model bound"):
        _ = result.typed
    assert result.bind(Invoice).typed.total == Decimal("1250.00")
    assert result.to_model(Invoice).vendor == "Acme"


def test_review_helpers_and_traces() -> None:
    result = _result()
    assert [f.path for f in result.needs_review_fields()] == ["vendor"]
    assert not result.succeeded
    stripped = result.without_traces()
    assert stripped.fields["total"].trace is None
    assert result.fields["total"].trace is not None


def test_explain_with_and_without_trace() -> None:
    result = _result()
    text = result.explain("total").text
    assert "status: VERIFIED" in text
    assert "'Total 1,250.00'" in text
    assert "r0:rule5:accept" in text
    assert "exact_quote +0.35" in text
    assert "review     budget" in result.explain("vendor").text
    assert "include_trace=True" in result.without_traces().explain("total").text
    with pytest.raises(KeyError, match="did you mean"):
        result.explain("tota")


def test_cost_and_check_summaries() -> None:
    report = CostReport.of(CostEntry("n", "m", Tier.CHEAP, "w", Usage(1, 1), Decimal("0.02"), 5))
    summary = CostSummary.of(report)
    assert summary.total_usd == Decimal("0.02")
    assert summary.by_tier == {1: Decimal("0.02")}
    check = CheckSummary.of(fail("total", "too big"))
    assert check.paths == ("total",)
    assert check.status == "fail"
