"""Synthetic salary slips: arithmetic, identifiers, words, rendering and ground truth."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

import pypdfium2 as pdfium
import pytest
from tools.synth.layouts.salary_layouts import SALARY_LAYOUTS
from tools.synth.ledger import EARLIEST_PERIOD_START, LATEST_PERIOD_END
from tools.synth.salary import SalarySlipGenerator
from tools.synth.truth import salary_slip_expected
from tools.synth.words import integer_in_words, rupees_in_words

SEEDS = range(40)


def _text(pdf: bytes) -> str:
    doc = pdfium.PdfDocument(pdf)
    try:
        return "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    finally:
        doc.close()


@pytest.mark.parametrize("seed", SEEDS)
def test_totals_add_up(seed: int) -> None:
    slip = SalarySlipGenerator(seed).generate()
    assert slip.gross_earnings == sum(e.amount for e in slip.earnings)
    assert slip.total_deductions == sum(d.amount for d in slip.deductions)
    assert slip.net_pay == slip.gross_earnings - slip.total_deductions
    assert slip.net_pay > 0
    assert all(line.amount > 0 for line in (*slip.earnings, *slip.deductions))


@pytest.mark.parametrize("seed", SEEDS)
def test_period_and_identifiers(seed: int) -> None:
    slip = SalarySlipGenerator(seed).generate()
    period = date(slip.pay_year, slip.pay_month, 1)
    assert EARLIEST_PERIOD_START <= period <= LATEST_PERIOD_END
    assert re.fullmatch(r"[A-Z]{3}P[A-Z]\d{4}[A-Z]", slip.pan)
    assert slip.uan is None or re.fullmatch(r"\d{12}", slip.uan)
    assert 0 <= slip.lop_days <= 2
    assert slip.date_of_joining < period


def test_same_seed_same_slip() -> None:
    a = salary_slip_expected(SalarySlipGenerator(4).generate())
    b = salary_slip_expected(SalarySlipGenerator(4).generate())
    assert a == b


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "Zero"),
        (15, "Fifteen"),
        (105, "One Hundred Five"),
        (45_612, "Forty Five Thousand Six Hundred Twelve"),
        (2_50_000, "Two Lakh Fifty Thousand"),
        (1_23_45_678, "One Crore Twenty Three Lakh Forty Five Thousand Six Hundred Seventy Eight"),
    ],
)
def test_integer_in_words(n: int, words: str) -> None:
    assert integer_in_words(n) == words


def test_rupees_in_words_with_paise() -> None:
    assert rupees_in_words(Decimal("45612.50")) == (
        "Rupees Forty Five Thousand Six Hundred Twelve and Fifty Paise Only"
    )


@pytest.mark.parametrize("layout_name", sorted(SALARY_LAYOUTS))
def test_every_amount_in_truth_is_printed(layout_name: str) -> None:
    layout = SALARY_LAYOUTS[layout_name]()
    for seed in range(5):
        slip = SalarySlipGenerator(seed).generate()
        text = _text(layout.render(slip))
        for line in (*slip.earnings, *slip.deductions):
            assert line.component in text
            assert layout.amount(line.amount) in text
        for total in (slip.gross_earnings, slip.total_deductions, slip.net_pay):
            assert layout.amount(total) in text
        assert slip.employee_name in text
        assert slip.pan in text
        assert rupees_in_words(slip.net_pay) in text
        assert "SYNTHETIC TEST DOCUMENT" in text


@pytest.mark.parametrize("layout_name", sorted(SALARY_LAYOUTS))
def test_salary_render_is_deterministic(layout_name: str) -> None:
    slip = SalarySlipGenerator(8).generate()
    layout = SALARY_LAYOUTS[layout_name]()
    assert layout.render(slip) == layout.render(slip)


def test_expected_json_shape() -> None:
    expected = salary_slip_expected(SalarySlipGenerator(2).generate())
    assert set(expected) == {
        "employer_name", "employee_name", "employee_id", "pan", "uan", "pay_period",
        "earnings", "deductions", "gross_earnings", "total_deductions", "net_pay",
        "bank_account_last4",
    }  # fmt: skip
    assert set(expected["pay_period"]) == {"month", "year"}
    assert re.fullmatch(r"\d+\.\d{2}", expected["net_pay"])
