"""Coercion, validation rules, quote location, value matching and evidence verification."""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bhoot.domain.candidate import CandidateValue
from bhoot.domain.checks import CheckResult, CheckStatus
from bhoot.domain.document import BBox, CharBox, Document, Page, PageKind, Word
from bhoot.domain.evidence import Evidence, EvidenceSource
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.locales import Locale
from bhoot.domain.modes import Tier
from bhoot.domain.rules import RuleContext, RuleScope, fail, pass_, rule
from bhoot.domain.schema import Criticality, FieldPath, FieldSpec
from bhoot.engine.data import assemble
from bhoot.engine.normalize.coercer import CoerceContext, TypeCoercer
from bhoot.engine.normalize.conventions import DocConventions
from bhoot.engine.validation import RuleRegistry, ValidatorChain
from bhoot.engine.verification import (
    EvidenceVerifier,
    QuoteLocator,
    ValueComparator,
    ValueInQuoteMatcher,
)

CONV = DocConventions(dayfirst=True, number_grouping="indian")
CTX = CoerceContext(CONV, Locale())


def _spec(path: str, py_type: str = "Decimal", **kw: Any) -> FieldSpec:
    json_type = {"Decimal": "number", "date": "date", "int": "integer", "bool": "boolean"}.get(
        py_type, "string"
    )
    return FieldSpec(
        FieldPath(path.replace("[0]", "[*]").replace("[1]", "[*]")), json_type, py_type, **kw
    )


def _cand(path: str, raw: Any, *, value: Any = None, quote: str | None = None, page: int | None = 0,
          tier: Tier = Tier.CHEAP) -> CandidateValue:  # fmt: skip
    c = CandidateValue(FieldPath(path), raw, "r0-x", 0, tier, quote=quote, page=page, model="m")
    return c.typed(value) if value is not None or raw is None else c


def _state(path: str, py_type: str = "Decimal", **kw: Any) -> FieldState:
    spec_kw = {k: kw.pop(k) for k in ("required", "criticality", "enum") if k in kw}
    spec = _spec(path, py_type, **spec_kw)
    cand = _cand(path, kw.pop("raw", "x"), **kw)
    return FieldState(FieldPath(path), spec).with_candidate(cand)


# ---------------------------------------------------------------- domain helpers
def test_a_row_check_implicates_every_cell_in_the_row() -> None:
    row = CheckResult("r", CheckStatus.FAIL, paths=(FieldPath("transactions[4]"),))
    assert row.implicates(FieldPath("transactions[4].balance"))
    assert not row.implicates(FieldPath("transactions[5].balance"))
    pattern = CheckResult("r", CheckStatus.FAIL, paths=(FieldPath("transactions[*].debit"),))
    assert pattern.implicates(FieldPath("transactions[9].debit"))
    assert not pattern.implicates(FieldPath("transactions[9]"))


def test_assemble_builds_nested_data() -> None:
    data = assemble({"a.b": 1, "rows[1].x": 2, "rows[0].x": 3, "tags[0]": "t", "z": None})
    assert data == {"a": {"b": 1}, "rows": [{"x": 3}, {"x": 2}], "tags": ["t"], "z": None}
    assert assemble({"rows[2].x": 1}) == {"rows": [{}, {}, {"x": 1}]}


# ---------------------------------------------------------------- coercion
@pytest.mark.parametrize(
    ("py_type", "raw", "expected"),
    [
        ("Decimal", "₹2,07,815.11", Decimal("207815.11")),
        ("Decimal", "1,234.00 Dr", Decimal("-1234.00")),
        ("Decimal", "-", None),
        ("int", "12", 12),
        ("date", "03/04/2026", date(2026, 4, 3)),
        ("date", "03-Apr-2026", date(2026, 4, 3)),
        ("bool", "Yes", True),
        ("str", "  Aster   Bank Ltd ", "Aster Bank Ltd"),
    ],
)
def test_coercion(py_type: str, raw: str, expected: Any) -> None:
    out = TypeCoercer().coerce(_cand("f", raw), _spec("f", py_type), CTX)
    assert out.coercion_error is None
    assert out.value == expected


def test_coercion_failures_are_recorded_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    coercer = TypeCoercer()
    assert coercer.coerce(_cand("f", "1.234,56"), _spec("f"), CTX).coercion_error is not None
    assert coercer.coerce(_cand("f", "12.5"), _spec("f", "int"), CTX).coercion_error is not None
    enum_spec = _spec("t", "AccountType", enum=("savings", "current"))
    assert coercer.coerce(_cand("t", "SAVINGS"), enum_spec, CTX).value == "savings"
    assert coercer.coerce(_cand("t", "savngs"), enum_spec, CTX).value == "savings"
    assert coercer.coerce(_cand("t", "loan"), enum_spec, CTX).coercion_error is not None
    with caplog.at_level(logging.WARNING, logger="bhoot.coercer"):
        assert coercer.coerce(_cand("p", " 12% "), _spec("p", "Percent"), CTX).value == "12%"
    assert "no coercer" in caplog.text
    assert coercer.coerce(_cand("f", None), _spec("f"), CTX).value is None


# ---------------------------------------------------------------- validation
@rule("t.acct_digits", scope=RuleScope.FIELD, applies_to=("account",))
def acct_digits(ctx: RuleContext) -> list[CheckResult]:
    value = ctx.values.get("account")
    return (
        [pass_("account")]
        if value is None or str(value).isdigit()
        else [fail("account", "digits only")]
    )


@rule("t.debit_xor_credit", scope=RuleScope.ROW, applies_to=("rows[*]",))
def debit_xor_credit(ctx: RuleContext) -> list[CheckResult]:
    out = []
    for i, row in enumerate(ctx.rows("rows")):
        if (row.get("debit") is None) == (row.get("credit") is None):
            out.append(fail(f"rows[{i}]", "exactly one of debit/credit"))
    return out


@rule("t.closing", scope=RuleScope.DOCUMENT, applies_to=("opening", "closing", "rows[*]"))
def closing_matches(ctx: RuleContext) -> list[CheckResult]:
    total = ctx.values["opening"] + sum(
        (r.get("credit") or 0) - (r.get("debit") or 0) for r in ctx.rows("rows")
    )
    if total != ctx.values["closing"]:
        return [
            fail(
                ["closing", "opening"],
                f"opening + movements = {total}, not {ctx.values['closing']}",
            )
        ]
    return [pass_(["closing"])]


@rule("t.crashes", scope=RuleScope.FIELD, applies_to=("account",))
def crashes(ctx: RuleContext) -> list[CheckResult]:
    raise KeyError("bug in a pack rule")


RULES = RuleRegistry([acct_digits, debit_xor_credit, closing_matches, crashes])


def _fields(**overrides: FieldState) -> dict[str, FieldState]:
    states = {
        "account": _state("account", "str", raw="123", value="123", required=True),
        "opening": _state("opening", raw="100", value=Decimal("100")),
        "closing": _state("closing", raw="130", value=Decimal("130")),
        "rows[0].debit": _state("rows[0].debit", raw=None),
        "rows[0].credit": _state("rows[0].credit", raw="50", value=Decimal("50")),
        "rows[1].debit": _state("rows[1].debit", raw="20", value=Decimal("20")),
        "rows[1].credit": _state("rows[1].credit", raw=None),
    }
    states.update(overrides)
    return states


def _chain() -> ValidatorChain:
    pack_rules = [
        RULES.get(i) for i in ("t.acct_digits", "t.debit_xor_credit", "t.closing", "t.crashes")
    ]
    return ValidatorChain(pack_rules)


def test_clean_values_are_validated_and_a_crashing_rule_only_warns() -> None:
    result = _chain().run(_fields(), locale=Locale())
    assert {fs.status for fs in result.field_states.values()} == {FieldStatus.VALIDATED}
    account_checks = {c.check_id: c.status for c in result.field_states["account"].checks}
    assert account_checks["t.acct_digits"] is CheckStatus.PASS
    assert [c.status for c in result.doc_checks] == [CheckStatus.PASS]


def test_rule_failures_mark_the_implicated_fields() -> None:
    bad_row = _state("rows[1].credit", raw="5", value=Decimal("5"))  # now debit AND credit
    result = _chain().run(_fields(**{"rows[1].credit": bad_row}), locale=Locale())
    states = result.field_states
    assert states["rows[1].debit"].status is FieldStatus.FAILED_CHECKS
    assert states["rows[1].credit"].status is FieldStatus.FAILED_CHECKS
    assert states["rows[0].credit"].status is FieldStatus.VALIDATED
    assert states["closing"].status is FieldStatus.FAILED_CHECKS  # 100 + 50 + 5 - 20 != 130
    assert "not 130" in states["closing"].history[-1][2]  # type: ignore[operator]


def test_type_failures_skip_document_rules_and_required_fields_must_have_values() -> None:
    opening = FieldState(FieldPath("opening"), _spec("opening")).with_candidate(
        _cand("opening", "1.0,0").failed_coercion("Decimal: ambiguous")
    )
    empty_account = _state("account", "str", raw=None, required=True)
    result = _chain().run(_fields(opening=opening, account=empty_account), locale=Locale())
    states = result.field_states
    assert states["opening"].status is FieldStatus.FAILED_CHECKS
    assert states["opening"].checks[0].check_id == "type.Decimal"
    assert states["account"].status is FieldStatus.FAILED_CHECKS
    assert states["account"].checks[0].check_id == "required"
    assert result.doc_checks[0].status is CheckStatus.SKIP


def test_a_later_failure_reopens_a_verified_field() -> None:
    first = _chain().run(_fields(), locale=Locale()).field_states
    verified = {p: fs.transition(FieldStatus.VERIFIED, "grounded") for p, fs in first.items()}
    repaired = _state("closing", raw="999", value=Decimal("999"))
    result = _chain().run({**verified, "closing": repaired}, locale=Locale())
    assert result.field_states["opening"].status is FieldStatus.FAILED_CHECKS  # was VERIFIED
    assert result.field_states["rows[0].credit"].status is FieldStatus.VERIFIED


# ---------------------------------------------------------------- quote location
def _page(i: int, text: str, kind: PageKind = PageKind.TEXT, **kw: Any) -> Page:
    chars = tuple(CharBox(ch, BBox(k * 5, 10, k * 5 + 5, 20), k) for k, ch in enumerate(text))
    return Page(i, 595, 842, kind, text, chars=chars, **kw)


def _doc(*pages: Page) -> Document:
    return Document("d", "s", "application/pdf", pages)


LOCATOR = QuoteLocator()


def test_locator_exact_normalised_fuzzy_and_neighbours() -> None:
    page = _page(0, "Opening 100.00\nClosing   Balance “1,250.00”")
    exact = LOCATOR.locate("Opening 100.00", page, threshold=90)
    assert exact is not None
    assert (exact.start, exact.end, exact.score) == (0, 14, 100.0)
    normal = LOCATOR.locate('Closing Balance "1,250.00"', page, threshold=90)
    assert normal is not None
    assert normal.score == 99.0
    assert page.text[normal.start : normal.end].startswith("Closing")
    fuzzy = LOCATOR.locate("Closing Balance 1,25O.00", page, threshold=85)
    assert fuzzy is not None
    assert 85 <= fuzzy.score < 99
    assert LOCATOR.locate("abc", _page(0, "xabdx"), threshold=50) is None  # too short to fuzz
    doc = _doc(_page(0, "first"), _page(1, "second page total 42.00"), _page(2, "third"))
    found = LOCATOR.locate_in("total 42.00", doc, 2, threshold=90)  # model said page 3
    assert found is not None
    assert found.page == 1
    assert LOCATOR.locate_in("total 42.00", doc, None, threshold=90) is not None


@given(
    st.text(alphabet="abcdefgh ,.0123456789", min_size=10, max_size=200),
    st.integers(min_value=0, max_value=190),
    st.integers(min_value=4, max_value=40),
)
def test_located_spans_always_point_at_the_quote(text: str, start: int, length: int) -> None:
    quote = text[start : start + length]
    if len(quote.strip()) < 4:
        return
    span = LOCATOR.locate(quote, _page(0, text), threshold=90)
    assert span is not None
    assert text[span.start : span.end] == quote


@given(
    st.lists(
        st.sampled_from(["closing", "balance", "1,250.00", "Dr", "total"]), min_size=2, max_size=8
    )
)
def test_whitespace_differences_still_locate(words: list[str]) -> None:
    text = "   ".join(words)
    span = LOCATOR.locate(" ".join(words), _page(0, text), threshold=90)
    assert span is not None
    assert span.score >= 99
    assert " ".join(text[span.start : span.end].split()) == " ".join(words)


# ---------------------------------------------------------------- matching
def test_value_in_quote_and_comparison() -> None:
    m = ValueInQuoteMatcher()
    amount = _spec("a")
    assert m.matches(Decimal("-1234"), "Withdrawal 1,234.00 Dr on 03/04", amount, CONV)
    assert not m.matches(Decimal("1243"), "Withdrawal 1,234.00 Dr", amount, CONV)
    assert m.matches(date(2026, 4, 3), "Txn 03-Apr-2026 UPI", _spec("d", "date"), CONV)
    assert m.matches(date(2026, 4, 3), "Txn 03/04/2026 UPI", _spec("d", "date"), CONV)
    assert not m.matches(date(2026, 3, 4), "Txn 03/04/2026 UPI", _spec("d", "date"), CONV)
    assert m.matches("Aster Bank Ltd", "ASTER BANK LTD, Pune", _spec("n", "str"), CONV)
    assert m.matches(None, None, amount, CONV)
    assert not m.matches("x", None, _spec("n", "str"), CONV)

    c = ValueComparator()
    assert c.equal(Decimal("1250.004"), Decimal("1250.00"))
    assert not c.equal(Decimal("1250.01"), Decimal("1250.00"))
    assert c.equal("Aster Bank Ltd", "aster  bank ltd.")
    assert not c.equal("savings", "current", _spec("t", "T", enum=("savings", "current")))
    assert c.equal(None, None)
    assert not c.equal(None, Decimal(0))


# ---------------------------------------------------------------- verification
def _validated(
    path: str, value: Any, quote: str | None, page: int | None, tier: Tier = Tier.CHEAP
) -> FieldState:
    fs = FieldState(FieldPath(path), _spec(path)).with_candidate(
        _cand(path, str(value), value=value, quote=quote, page=page, tier=tier)
    )
    return fs.transition(FieldStatus.VALIDATED)


def test_verifier_grounds_values_on_the_page() -> None:
    ocr_page = _page(
        1, "Closing Balance 1,250.00", PageKind.OCR_TEXT, text_source="ocr",
        words=(Word("Closing", BBox(0, 10, 35, 20), 0.99), Word("1,250.00", BBox(80, 10, 120, 20), 0.71)),
    )  # fmt: skip
    doc = _doc(_page(0, "Opening Balance 100.00"), ocr_page, _page(2, "", PageKind.SCANNED))
    states = {
        "opening": _validated("opening", Decimal("100.00"), "Opening Balance 100.00", 0),
        "closing": _validated("closing", Decimal("1250.00"), "Closing Balance 1,250.00", 1),
        "wrong": _validated("wrong", Decimal("1520.00"), "Closing Balance 1,250.00", 1),
        "missing": _validated("missing", Decimal("5"), "Total 5.00", 0),
        "noquote": _validated("noquote", Decimal("5"), None, 0),
        "scanned": _validated("scanned", Decimal("77"), "Fee 77.00", 2),
        "code": _validated("code", Decimal("1"), None, 0, Tier.CODE),
    }
    out = EvidenceVerifier().verify(states, doc, CONV, threshold=90)
    status = {p: fs.status for p, fs in out.items()}
    assert status == {
        "opening": FieldStatus.VERIFIED,
        "closing": FieldStatus.VERIFIED,
        "wrong": FieldStatus.UNGROUNDED,
        "missing": FieldStatus.UNGROUNDED,
        "noquote": FieldStatus.UNGROUNDED,
        "scanned": FieldStatus.VERIFIED,
        "code": FieldStatus.VERIFIED,
    }
    ev = out["opening"].current.evidence  # type: ignore[union-attr]
    assert ev is not None
    assert ev.char_span == (0, 22)
    assert ev.bbox is not None
    assert ev.grounded
    closing = out["closing"].current.evidence  # type: ignore[union-attr]
    assert closing is not None
    assert closing.ocr_min_confidence == pytest.approx(0.71)
    scanned = out["scanned"].current.evidence  # type: ignore[union-attr]
    assert scanned is not None
    assert not scanned.grounded
    assert scanned.match_score == 0.0
    assert "not in the text quoted on page 2" in out["wrong"].history[-1][2]  # type: ignore[operator]
    assert Evidence(0, "q", EvidenceSource.CODE).grounded
    assert out["code"].spec.criticality is Criticality.NORMAL
