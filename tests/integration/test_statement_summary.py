"""Phase 1 exit test: the whole engine on synthetic bank statements from ``evals/``.

The model is a "perfect reader": it only sees the page text the engine sends it and answers
with the values exactly as printed, quoting the line it read, like an ideal model would.
Everything else is real: parsing, page planning, per-document conventions (day-first dates,
Indian vs international grouping), coercion, a document rule, quote grounding and the result.
"""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from bhoot import (
    Bhoot,
    CheckResult,
    JobStatus,
    RuleContext,
    RuleScope,
    Schema,
    field,
    pass_,
    rule,
    warn,
)
from bhoot.adapters.llm.fake import FakeLLMClient
from bhoot.config.models import ModelSpec, ModelTierConfig
from bhoot.config.settings import BhootSettings
from bhoot.domain.pack import DocPack, SectionSpec
from bhoot.domain.schema_introspect import introspect
from bhoot.engine.prompting import DEFAULT_PROMPT
from bhoot.ports.llm import LLMRequest

EVALS = Path(__file__).resolve().parents[2] / "evals" / "bank_statement"
CASES = ["case_0001", "case_0002", "case_0003", "case_0004", "case_0005"]


class Period(Schema):
    from_date: date = field(criticality="important")
    to_date: date = field(criticality="important")


class StatementSummary(Schema):
    account_number: str = field(criticality="critical")
    account_holder: str = field(criticality="important")
    ifsc: str | None = field(default=None)
    period: Period
    opening_balance: Decimal = field(criticality="critical")
    closing_balance: Decimal = field(criticality="critical")


@rule(
    "test.balances_present",
    scope=RuleScope.DOCUMENT,
    applies_to=("opening_balance", "closing_balance"),
)
def balances_present(ctx: RuleContext) -> list[CheckResult]:
    if ctx.values.get("opening_balance") is None or ctx.values.get("closing_balance") is None:
        return [warn(["closing_balance"], "a balance is missing")]
    return [pass_(["opening_balance", "closing_balance"])]


PACK = DocPack(
    name="test.statement_summary",
    version=1,
    schema=introspect(StatementSummary),
    prompt=DEFAULT_PROMPT,
    sections=(SectionSpec(name="summary", paths=("*",)),),
    rules=(balances_present,),
)

_PAGE = re.compile(r'<page n="(\d+)">\n(.*?)\n</page>', re.DOTALL)
_PATTERNS = {
    "account_number": r"Account Number:\s*(\S+)",
    "account_holder": r"Account Holder:\s*([^\r\n]+)",
    "ifsc": r"IFSC:\s*(\S+)",
    "opening_balance": r"Opening Balance (?:INR|Rs\.)\s*([\d,.]+)",
    "closing_balance": r"Closing Balance (?:INR|Rs\.)\s*([\d,.]+)",
}


def _read(pages: list[tuple[int, str]], pattern: str, group: int = 1) -> dict[str, Any]:
    for number, text in pages:
        for line in text.splitlines():
            match = re.search(pattern, line)
            if match:
                return {"value": match.group(group), "quote": line.strip(), "page": number}
    return {"value": None, "quote": None, "page": None}


def perfect_reader(request: LLMRequest) -> dict[str, Any]:
    text = "".join(p.text or "" for p in request.messages[-1].parts)
    pages = [(int(n), body.replace("<\\/", "</")) for n, body in _PAGE.findall(text)]
    answer = {name: _read(pages, pattern) for name, pattern in _PATTERNS.items()}
    period = r"Statement period:\s*(.+?) to (.+?)\s*$"
    answer["period"] = {"from_date": _read(pages, period, 1), "to_date": _read(pages, period, 2)}
    return answer


def _bhoot() -> tuple[Bhoot, FakeLLMClient]:
    fake = FakeLLMClient()
    fake.when().respond(perfect_reader)
    models = ModelTierConfig(
        cheap=ModelSpec(model="openai/cheap"), strong=ModelSpec(model="openai/strong")
    )
    return Bhoot(settings=BhootSettings(models=models), llm=fake), fake


@pytest.mark.parametrize("case", CASES)
def test_statement_summary_matches_the_truth(case: str) -> None:
    folder = EVALS / case
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    bh, fake = _bhoot()
    with bh:
        result = bh.extract(
            folder / "document.pdf", schema=PACK, mode="balanced", include_trace=True
        )
    assert result.status is JobStatus.SUCCEEDED, result.review_reasons
    data = result.data
    assert data["account_number"] == expected["account_number"]
    assert data["account_holder"].casefold() == expected["account_holder"].casefold()
    assert data["ifsc"] == expected["ifsc"]
    assert data["opening_balance"] == Decimal(expected["opening_balance"])
    assert data["closing_balance"] == Decimal(expected["closing_balance"])
    assert data["period"]["from_date"] == date.fromisoformat(expected["period"]["from_date"])
    assert data["period"]["to_date"] == date.fromisoformat(expected["period"]["to_date"])
    for name in ("account_number", "opening_balance", "closing_balance", "period.to_date"):
        fr = result.fields[name]
        assert fr.status == "verified"
        assert fr.bbox is not None  # grounded on the page, not just claimed
        assert fr.confidence >= 0.9
    assert all(c.status == "pass" for c in result.checks)
    assert result.cost.llm_calls == len(fake.calls)  # one call per page
    assert result.document is not None
    assert len(fake.calls) == result.document.pages
