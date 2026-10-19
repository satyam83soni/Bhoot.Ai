"""Schema introspection, evidence envelopes, provider dialects and unwrapping answers."""

from __future__ import annotations

import datetime as dt
import enum
from decimal import Decimal
from typing import Any, Literal

import pytest

from bhoot.domain.candidate import CandidateRole
from bhoot.domain.errors import ResponseParseError
from bhoot.domain.modes import Tier
from bhoot.domain.schema import Criticality, FieldPath, Granularity
from bhoot.domain.schema_dsl import Schema, field
from bhoot.domain.schema_introspect import SchemaIntrospectionError, introspect
from bhoot.domain.work import WorkItem, WorkKind
from bhoot.engine.schema_compiler import (
    AnthropicDialect,
    EnvelopeUnwrapper,
    EvidenceSchemaBuilder,
    GeminiDialect,
    OpenAIStrictDialect,
    PromptOnlyDialect,
    dialect_for,
    parse_answer,
)
from bhoot.engine.schema_compiler.dialects import format_name
from bhoot.ports.llm import LLMResponse, ModelCapabilities


class AccountType(enum.Enum):
    SAVINGS = "savings"
    CURRENT = "current"


class Period(Schema):
    from_date: dt.date = field(criticality="important")
    to_date: dt.date = field(criticality="important")


class Transaction(Schema):
    date: dt.date = field(criticality="important")
    description: str = field()
    debit: Decimal | None = field(criticality="critical", default=None)
    credit: Decimal | None = field(criticality="critical", default=None)
    balance: Decimal = field(criticality="critical")


class Statement(Schema):
    account_number: str = field(criticality="critical", description="Account number")
    account_type: AccountType | None = field(default=None)
    currency: Literal["INR", "USD"] | None = None
    period: Period
    closing_balance: Decimal = field(criticality="critical")
    tags: list[str] = field(default_factory=list)
    transactions: list[Transaction] = field(granularity="row")


SCHEMA = introspect(Statement)


# ---------------------------------------------------------------- introspection
def test_introspection_builds_specs_by_pattern_path() -> None:
    f = SCHEMA.fields
    assert SCHEMA.name == "statement"
    assert list(f)[:3] == ["account_number", "account_type", "currency"]
    assert f["account_number"].criticality is Criticality.CRITICAL
    assert f["account_number"].required
    assert f["account_number"].description == "Account number"
    assert not f["account_type"].required
    assert f["account_type"].enum == ("savings", "current")
    assert f["account_type"].py_type == "AccountType"
    assert f["currency"].enum == ("INR", "USD")
    assert f["period.from_date"].json_type == "date"
    assert f["period.from_date"].fmt == "date"
    assert f["closing_balance"].py_type == "Decimal"
    assert f["tags[*]"].granularity is Granularity.LEAF
    assert f["transactions"].granularity is Granularity.ROW
    assert f["transactions[*].debit"].granularity is Granularity.ROW
    assert not f["transactions[*].debit"].required
    assert f["transactions[*].balance"].required
    assert SCHEMA.spec_for(FieldPath("transactions[7].balance")).criticality is Criticality.CRITICAL
    assert "$defs" not in str(SCHEMA.json_schema)


def test_schema_hash_changes_only_when_the_schema_does() -> None:
    assert introspect(Statement).content_hash == SCHEMA.content_hash

    class Looser(Schema):
        account_number: str = field(criticality="normal")

    class Stricter(Schema):
        account_number: str = field(criticality="critical")

    assert introspect(Looser).content_hash != introspect(Stricter).content_hash


def test_unsupported_types_are_explained() -> None:
    class WithInstant(Schema):
        at: dt.datetime

    class WithUnion(Schema):
        x: int | str

    with pytest.raises(SchemaIntrospectionError, match="use date"):
        introspect(WithInstant)
    with pytest.raises(SchemaIntrospectionError, match="X \\| None"):
        introspect(WithUnion)


# ---------------------------------------------------------------- envelopes
def _strict_ready(node: Any) -> None:
    """What strict structured output requires of every object in the schema."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])
        for value in node.values():
            _strict_ready(value)
    elif isinstance(node, list):
        for value in node:
            _strict_ready(value)


def test_full_envelope_is_strict_ready_with_evidence_everywhere() -> None:
    env = EvidenceSchemaBuilder().build(SCHEMA)
    _strict_ready(env)
    props = env["properties"]
    leaf = props["closing_balance"]["properties"]
    assert set(leaf) == {"value", "quote", "page"}
    assert leaf["value"]["type"] == ["string", "null"]
    assert "exactly as printed" in leaf["value"]["description"].lower()
    assert "One of: savings, current" in props["account_type"]["properties"]["value"]["description"]
    assert set(props["period"]["properties"]) == {"from_date", "to_date"}
    row = props["transactions"]["items"]["properties"]
    assert list(row) == ["date", "description", "debit", "credit", "balance", "_quote", "_page"]
    assert row["debit"]["type"] == ["string", "null"]
    assert set(props["tags"]["items"]["properties"]) == {"value", "quote", "page"}


def test_subset_envelopes_ask_only_for_failing_fields_and_whole_rows() -> None:
    builder = EvidenceSchemaBuilder()
    one = builder.build(SCHEMA, only=[FieldPath("closing_balance")])
    assert list(one["properties"]) == ["closing_balance"]
    nested = builder.build(SCHEMA, only=[FieldPath("period.to_date")])
    assert list(nested["properties"]["period"]["properties"]) == ["to_date"]
    rows = builder.build(SCHEMA, only=[FieldPath("transactions[12].debit")])
    assert list(rows["properties"]) == ["transactions"]
    assert "balance" in rows["properties"]["transactions"]["items"]["properties"]
    with pytest.raises(ValueError, match="no fields"):
        builder.build(SCHEMA, only=[FieldPath("nope")])


# ---------------------------------------------------------------- dialects
def test_dialects_shape_the_same_envelope_per_provider() -> None:
    env = EvidenceSchemaBuilder().build(SCHEMA)
    openai = OpenAIStrictDialect().adapt(env, "pack:lending.bank statement")
    assert openai is not None
    assert openai.strict
    assert openai.name == "pack_lending_bank_statement"
    assert dict(openai.json_schema) == env

    anthropic = AnthropicDialect().adapt(env, "x")
    assert anthropic is not None
    assert not anthropic.strict

    gemini = GeminiDialect().adapt(env, "x")
    assert gemini is not None
    text = str(gemini.json_schema)
    assert "additionalProperties" not in text
    value = gemini.json_schema["properties"]["closing_balance"]["properties"]["value"]
    assert value["anyOf"] == [{"type": "string"}, {"type": "null"}]

    prompt = PromptOnlyDialect()
    assert prompt.adapt(env, "x") is None
    assert '"closing_balance"' in prompt.instructions(env)


def test_unknown_dialect_falls_back_to_prompt_only_with_a_warning() -> None:
    caps = ModelCapabilities(
        vision=False, json_schema=False, strict_schema=False, dialect="mystery"
    )
    dialect, warning = dialect_for(caps)
    assert dialect.name == "prompt_only"
    assert warning is not None
    assert "mystery" in warning
    known = ModelCapabilities(vision=True, json_schema=True, strict_schema=True, dialect="gemini")
    chosen, no_warning = dialect_for(known)
    assert isinstance(chosen, GeminiDialect)
    assert no_warning is None
    assert format_name("!!!") == "_"


# ---------------------------------------------------------------- unwrapping
def _item(kind: WorkKind = WorkKind.EXTRACT, window: tuple[int, int] | None = None) -> WorkItem:
    return WorkItem(
        "r0-all-p1-2-extract-0", 0, kind, "all", (0, 1), Tier.CHEAP, "cheap", row_window=window
    )


ANSWER = {
    "account_number": {"value": " 718347936524 ", "quote": "A/c No: 718347936524", "page": 1},
    "account_type": {"value": None, "quote": None, "page": None},
    "currency": {"value": "INR", "quote": "Currency INR", "page": "1"},
    "period": {
        "from_date": {"value": "01/04/2026", "quote": "From 01/04/2026", "page": 1},
        "to_date": {"value": "", "quote": None, "page": None},
    },
    "closing_balance": {"value": "2,07,815.11", "quote": "Closing 2,07,815.11", "page": 2},
    "tags": [{"value": "salary", "quote": "salary", "page": 1}],
    "transactions": [
        {"date": "03/04/2026", "description": "UPI/DR/317", "debit": "306.61", "credit": None,
         "balance": "8,785.11", "_quote": "03/04/2026 UPI/DR/317 306.61 8,785.11", "_page": 1},
        {"date": None, "description": None, "debit": None, "credit": None, "balance": None,
         "_quote": None, "_page": None},
        {"date": "05/04/2026", "description": "UPI/DR/715", "debit": "165.41", "credit": None,
         "balance": "8,619.70", "_quote": "05/04/2026 UPI/DR/715 165.41 8,619.70", "_page": 2},
    ],
}  # fmt: skip


def test_unwrap_turns_an_answer_into_candidates_and_nulls() -> None:
    out = EnvelopeUnwrapper().unwrap(ANSWER, schema=SCHEMA, item=_item(), model="openai/mini")
    by_path = {c.path.raw: c for c in out.candidates}
    acct = by_path["account_number"]
    assert acct.raw == "718347936524"
    assert acct.page == 0
    assert acct.quote == "A/c No: 718347936524"
    assert acct.role is CandidateRole.PRIMARY
    assert acct.source_kind is WorkKind.EXTRACT
    assert by_path["currency"].page == 0
    assert by_path["tags[0]"].raw == "salary"
    assert {p.raw for p in out.nulls} == {"account_type", "period.to_date"}
    # the empty row is dropped, so the second real row gets index 1
    assert by_path["transactions[1].balance"].raw == "8,619.70"
    assert by_path["transactions[1].balance"].page == 1
    assert by_path["transactions[0].credit"].raw is None  # a credit-less row still has the column
    assert by_path["transactions[0].debit"].quote.startswith("03/04/2026")  # type: ignore[union-attr]
    assert out.row_counts == {"transactions": 2}
    for candidate in out.candidates:
        SCHEMA.spec_for(candidate.path)  # every path is a real field


def test_repair_windows_and_cross_checks() -> None:
    rows = {"transactions": ANSWER["transactions"][:1]}
    repair = EnvelopeUnwrapper().unwrap(
        rows, schema=SCHEMA, item=_item(WorkKind.REPAIR, (40, 41)), model="m",
        only=(FieldPath("transactions[40].debit"),),
    )  # fmt: skip
    assert {c.path.raw for c in repair.candidates} >= {"transactions[40].debit"}
    cross = EnvelopeUnwrapper().unwrap(
        {"closing_balance": {"value": "1", "quote": "1", "page": 1}},
        schema=SCHEMA, item=_item(WorkKind.CROSSCHECK), model="m",
        only=(FieldPath("closing_balance"),),
    )  # fmt: skip
    assert cross.candidates[0].role is CandidateRole.ALTERNATIVE


def test_parse_answer_is_lenient_about_wrapping_but_not_garbage() -> None:
    fenced = LLMResponse(content='```json\n{"a": 1}\n```', model="m")
    assert parse_answer(fenced) == {"a": 1}
    prose = LLMResponse(content='Here you go: {"a": {"b": 2}} hope it helps', model="m")
    assert parse_answer(prose) == {"a": {"b": 2}}
    assert parse_answer(LLMResponse(content="", model="m", parsed={"x": 1})) == {"x": 1}
    with pytest.raises(ResponseParseError):
        parse_answer(LLMResponse(content="I could not read the page", model="m"))
    with pytest.raises(ResponseParseError):
        parse_answer(LLMResponse(content="[1, 2]", model="m"))
