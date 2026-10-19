from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bhoot.domain.locales import Locale
from bhoot.engine.normalize.conventions import detect_conventions
from bhoot.engine.normalize.dates import DateParseError, is_ambiguous, parse_date
from bhoot.engine.normalize.text import (
    normalize_for_match,
    normalize_with_index_map,
    strip_currency,
)

D = date(2025, 4, 3)


@pytest.mark.parametrize(
    "raw",
    ["03/04/2025", "03-04-2025", "03.04.2025", "03/04/25", "3-Apr-2025", "03 Apr 2025",
     "3 April 2025", "2025-04-03", "Apr 3, 2025", "3rd April 2025", "03-Apr-25", "  03/04/2025 "],
)  # fmt: skip
def test_parse_date_dayfirst(raw: str) -> None:
    assert parse_date(raw, dayfirst=True) == D


def test_monthfirst_reading() -> None:
    assert parse_date("04/03/2025", dayfirst=False) == D
    assert parse_date("25/04/2025", dayfirst=False) == date(2025, 4, 25)  # only one valid reading


@pytest.mark.parametrize("raw", [None, "", "-", "N/A"])
def test_empty_dates(raw: str | None) -> None:
    assert parse_date(raw) is None


@pytest.mark.parametrize("raw", ["31/02/2025", "hello", "2025", "32-01-2025"])
def test_invalid_dates_raise(raw: str) -> None:
    with pytest.raises(DateParseError):
        parse_date(raw)


def test_ambiguity() -> None:
    assert is_ambiguous("03/04/2025")
    assert not is_ambiguous("25/04/2025")
    assert not is_ambiguous("04/04/2025")  # same either way
    assert not is_ambiguous("3 April 2025")


@given(st.dates(min_value=date(1950, 1, 1), max_value=date(2049, 12, 31)))
def test_dates_round_trip_in_common_formats(d: date) -> None:
    for fmt in ("%d/%m/%Y", "%d-%b-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %B %Y"):
        assert parse_date(d.strftime(fmt), dayfirst=True) == d


# ---------------------------------------------------------------- conventions
def test_conventions_from_document_evidence() -> None:
    conv = detect_conventions(
        ["03/04/2025", "25/04/2025", "28/04/2025"], ["12,34,567.00"], Locale()
    )
    assert conv.dayfirst
    assert conv.number_grouping == "indian"
    assert conv.date_evidence == 2
    us = detect_conventions(
        ["04/25/2025", "04/03/2025"], ["1,234,567.00", "2,345,678.00"], Locale()
    )
    assert not us.dayfirst
    assert us.number_grouping == "international"


def test_one_comma_can_still_decide_the_grouping() -> None:
    # "117,861.77" can only be international (Indian would print 1,17,861.77)
    assert (
        detect_conventions([], ["117,861.77", "93,142.19"], Locale()).number_grouping
        == "international"
    )
    # "12,345.00" reads the same both ways: no evidence, so values are read as printed
    conv = detect_conventions([], ["12,345.00", "500.00"], Locale())
    assert conv.number_grouping == "auto"
    assert conv.number_evidence == 0
    tie = detect_conventions([], ["1,17,861.77", "117,861.77"], Locale())
    assert tie.number_grouping == "auto"


def test_conventions_fall_back_to_locale() -> None:
    conv = detect_conventions(
        ["03/04/2025"], ["1,234.00"], Locale(dayfirst=False, number_grouping="international")
    )
    assert not conv.dayfirst
    assert conv.number_grouping == "international"
    assert conv.date_evidence == 0


# ---------------------------------------------------------------- text
def test_normalize_for_match() -> None:
    assert (
        normalize_for_match("  Closing\u00a0Balance \u2014  48,250.75 ")
        == "closing balance - 48,250.75"
    )
    assert normalize_for_match("\u201cQuoted\u201d\u200b text") == '"quoted" text'
    assert normalize_for_match("\ufb01nal") == "final"


@given(st.text(max_size=60))
def test_index_map_points_back_into_the_source(text: str) -> None:
    norm, back = normalize_with_index_map(text)
    assert len(norm) == len(back)
    assert back == sorted(back)
    assert all(0 <= i < len(text) for i in back)
    assert not norm.startswith(" ")


def test_index_map_recovers_original_span() -> None:
    source = "Closing  Balance   48,250.75"
    norm, back = normalize_with_index_map(source)
    start = norm.index("48,250.75")
    end = start + len("48,250.75")
    assert source[back[start] : back[end - 1] + 1] == "48,250.75"


def test_strip_currency() -> None:
    assert strip_currency("rs. 500") == "500"
    assert strip_currency("₹ 1,000 inr") == "1,000"
