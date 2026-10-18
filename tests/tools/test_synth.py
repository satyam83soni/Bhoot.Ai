"""Synthetic data generator: ledger invariants, formatting, rendering and ground truth."""

from __future__ import annotations

import re
from decimal import Decimal

import pypdfium2 as pdfium
import pytest
from tools.synth.layouts import LAYOUTS
from tools.synth.layouts.base import FONT, format_amount, wrap_text
from tools.synth.ledger import (
    EARLIEST_PERIOD_START,
    LATEST_PERIOD_END,
    MIN_BALANCE,
    LedgerGenerator,
)
from tools.synth.truth import bank_statement_expected

SEEDS = range(40)


def _pdf_text(pdf: bytes) -> tuple[str, int]:
    doc = pdfium.PdfDocument(pdf)
    try:
        pages = [doc[i].get_textpage().get_text_range() for i in range(len(doc))]
        return "\n".join(pages), len(doc)
    finally:
        doc.close()


# ---------------------------------------------------------------- ledger invariants
@pytest.mark.parametrize("seed", SEEDS)
def test_running_balance_chains(seed: int) -> None:
    ledger = LedgerGenerator(seed).generate()
    previous = ledger.opening_balance
    for t in ledger.transactions:
        assert (t.debit is None) != (t.credit is None), "exactly one of debit/credit"
        expected = previous - (t.debit or Decimal(0)) + (t.credit or Decimal(0))
        assert t.balance == expected
        assert t.balance >= MIN_BALANCE
        previous = t.balance


@pytest.mark.parametrize("seed", SEEDS)
def test_opening_plus_credits_minus_debits_is_closing(seed: int) -> None:
    ledger = LedgerGenerator(seed).generate()
    assert (
        ledger.opening_balance + ledger.total_credits - ledger.total_debits
        == ledger.closing_balance
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_dates_ordered_and_within_period(seed: int) -> None:
    ledger = LedgerGenerator(seed).generate()
    dates = [t.date for t in ledger.transactions]
    assert dates == sorted(dates)
    assert all(ledger.period_from <= d <= ledger.period_to for d in dates)
    assert all(t.value_date >= t.date for t in ledger.transactions)
    assert ledger.period_from >= EARLIEST_PERIOD_START
    assert ledger.period_to <= LATEST_PERIOD_END


def test_same_seed_same_ledger() -> None:
    assert bank_statement_expected(LedgerGenerator(3).generate()) == bank_statement_expected(
        LedgerGenerator(3).generate()
    )


def test_identifiers_have_valid_formats() -> None:
    for seed in SEEDS:
        account = LedgerGenerator(seed).generate(months=1).account
        assert re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", account.ifsc)
        assert re.fullmatch(r"[1-9]\d{10,15}", account.account_number)


# ---------------------------------------------------------------- formatting
@pytest.mark.parametrize(
    ("value", "grouping", "expected"),
    [
        ("2500000.00", "indian", "25,00,000.00"),
        ("123456789.5", "indian", "12,34,56,789.50"),
        ("999.00", "indian", "999.00"),
        ("1000.00", "indian", "1,000.00"),
        ("2500000.50", "international", "2,500,000.50"),
        ("0.00", "international", "0.00"),
    ],
)
def test_format_amount(value: str, grouping: str, expected: str) -> None:
    assert format_amount(Decimal(value), grouping) == expected  # type: ignore[arg-type]


def test_wrap_text_fits_and_keeps_all_characters() -> None:
    text = "UPI/DR/412345678901/QUICKMART GROCERY/quickmartgr42@upi/PAYMENT"
    lines = wrap_text(text, 80, FONT, 7.5)
    assert len(lines) > 1
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


# ---------------------------------------------------------------- rendering + truth
@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_every_balance_in_truth_is_printed(layout_name: str) -> None:
    ledger = LedgerGenerator(101).generate(months=3)
    layout = LAYOUTS[layout_name]()
    text, pages = _pdf_text(layout.render(ledger))
    compact = text.replace("\r", "")

    assert ledger.account.account_number in compact
    assert "SYNTHETIC TEST DOCUMENT" in compact
    assert f"Page {pages} of {pages}" in compact
    for t in ledger.transactions:
        assert format_amount(t.balance, layout.style.grouping) in compact, (layout_name, t)
    assert format_amount(ledger.closing_balance, layout.style.grouping) in compact


@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_render_is_deterministic(layout_name: str) -> None:
    ledger = LedgerGenerator(5).generate(months=1)
    layout = LAYOUTS[layout_name]()
    assert layout.render(ledger) == layout.render(ledger)


def test_long_statement_spans_several_pages() -> None:
    ledger = LedgerGenerator(11).generate(months=6)
    _, pages = _pdf_text(LAYOUTS["bank_a"]().render(ledger))
    assert len(ledger.transactions) > 100
    assert pages >= 3


def test_expected_json_shape() -> None:
    expected = bank_statement_expected(LedgerGenerator(9).generate(months=2))
    assert set(expected) == {
        "bank_name",
        "account_holder",
        "account_number",
        "ifsc",
        "account_type",
        "currency",
        "period",
        "opening_balance",
        "closing_balance",
        "transactions",
    }
    row = expected["transactions"][0]
    assert set(row) == {
        "date",
        "value_date",
        "description",
        "reference",
        "debit",
        "credit",
        "balance",
    }
    assert re.fullmatch(r"\d+\.\d{2}", row["balance"])
