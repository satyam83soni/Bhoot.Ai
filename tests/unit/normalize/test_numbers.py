from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bhoot.engine.normalize.numbers import (
    AmbiguousNumberError,
    MalformedNumberError,
    format_amount,
    parse_amount,
)


@pytest.mark.parametrize(
    ("raw", "value", "side"),
    [
        ("₹25,00,000.00", Decimal("2500000.00"), None),
        ("25,00,000", Decimal("2500000"), None),
        ("2,500,000.50", Decimal("2500000.50"), None),
        ("1,234.00 Dr", Decimal("-1234.00"), "debit"),
        ("1,234.00 Cr", Decimal("1234.00"), "credit"),
        ("1,234.00CR", Decimal("1234.00"), "credit"),
        ("(1,234.00)", Decimal("-1234.00"), None),
        ("1,234.00-", Decimal("-1234.00"), None),
        ("-500", Decimal("-500"), None),
        ("Rs. 500/-", Decimal("500"), None),
        ("INR 12,345.67", Decimal("12345.67"), None),
        ("1, 234 .50", Decimal("1234.50"), None),  # OCR spacing
        ("0.75", Decimal("0.75"), None),
        (".75", Decimal("0.75"), None),
        (1500, Decimal("1500"), None),
        (Decimal("2.5"), Decimal("2.5"), None),
    ],
)
def test_parse_amount(raw: object, value: Decimal, side: str | None) -> None:
    parsed = parse_amount(raw)  # type: ignore[arg-type]
    assert parsed.value == value
    assert parsed.side == side


@pytest.mark.parametrize("raw", [None, "", "-", "--", "NIL", "n/a", " - "])
def test_empty_values(raw: str | None) -> None:
    assert parse_amount(raw).value is None


@pytest.mark.parametrize("raw", ["1.234,56", "12.345,00"])
def test_european_format_is_ambiguous(raw: str) -> None:
    with pytest.raises(AmbiguousNumberError):
        parse_amount(raw)


@pytest.mark.parametrize(
    "raw", ["1,23,4567.00", "12,3456", "1,2,3", "abc", "1.2.3", "12a", "1,00,00,0"]
)
def test_malformed_numbers_are_refused(raw: str) -> None:
    with pytest.raises(MalformedNumberError):
        parse_amount(raw)


def test_explicit_grouping_policy() -> None:
    assert parse_amount("12,34,567.00", "indian").grouping == "indian"
    with pytest.raises(MalformedNumberError, match="indian"):
        parse_amount("1,234,567.00", "indian")
    with pytest.raises(MalformedNumberError, match="international"):
        parse_amount("12,34,567.00", "international")
    assert parse_amount("12,345.00", "indian").value == Decimal("12345.00")  # fits both


def test_detected_grouping() -> None:
    assert parse_amount("12,34,567").grouping == "indian"
    assert parse_amount("1,234,567").grouping == "international"
    assert parse_amount("1234567").grouping == "none"


amounts = st.decimals(
    min_value=0, max_value=10**12, places=2, allow_nan=False, allow_infinity=False
)


@given(amounts, st.sampled_from(["indian", "international"]))
def test_format_then_parse_round_trips(value: Decimal, style: str) -> None:
    text = format_amount(value, style)  # type: ignore[arg-type]
    assert parse_amount(text).value == value
    assert parse_amount(f"Rs. {text}/-").value == value
    assert parse_amount(f"({text})").value == -value


def test_format_amount_examples() -> None:
    assert format_amount(Decimal("2500000.00")) == "25,00,000.00"
    assert format_amount(Decimal("2500000.5"), "international") == "2,500,000.5"
    assert format_amount(Decimal("-999")) == "-999"
