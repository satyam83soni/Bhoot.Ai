"""Value-in-quote matching and value comparison.

A located quote is not enough: the model could quote the right line and still report a wrong
number. ``ValueInQuoteMatcher`` checks that the typed value really appears in the quote.
``ValueComparator`` decides whether two answers (cross-check, crop re-read) agree.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Any

from rapidfuzz import fuzz

from bhoot.domain.schema import FieldSpec
from bhoot.engine.normalize.conventions import DocConventions
from bhoot.engine.normalize.dates import parse_date
from bhoot.engine.normalize.numbers import AmountParseError, parse_amount
from bhoot.engine.normalize.text import normalize_for_match

AMOUNT_TOLERANCE = Decimal("0.005")
STR_IN_QUOTE_THRESHOLD = 92
STR_EQUAL_THRESHOLD = 95

_AMOUNT_TOKEN = re.compile(
    r"\(?-?(?:₹|rs\.?|inr)?\s*\d[\d,]*(?:\.\d+)?\s*(?:dr|cr)?\.?\)?", re.IGNORECASE
)
_DATE_TOKENS = (
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?[\s-][A-Za-z]{3,9},?[\s-]\d{2,4}\b"),
    re.compile(r"\b[A-Za-z]{3,9}\s\d{1,2},\s\d{4}\b"),
)


def _amounts(text: str) -> list[Decimal]:
    out = []
    for token in _AMOUNT_TOKEN.findall(text):
        try:
            value = parse_amount(token.strip()).value
        except AmountParseError:
            continue
        if value is not None:
            out.append(value)
    return out


def _dates(text: str, dayfirst: bool) -> list[date]:
    out = []
    for pattern in _DATE_TOKENS:
        for token in pattern.findall(text):
            try:
                parsed = parse_date(token, dayfirst=dayfirst)
            except ValueError:
                continue
            if parsed is not None:
                out.append(parsed)
    return out


def _numeric(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, Decimal)):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    return None


class ValueInQuoteMatcher:
    def matches(
        self, value: Any, quote: str | None, spec: FieldSpec, conventions: DocConventions
    ) -> bool:
        if value is None or isinstance(value, bool):
            return True  # nothing printed to find (null cells, yes/no flags)
        if not quote:
            return False
        number = _numeric(value)
        if number is not None:
            return any(
                abs(abs(found) - abs(number)) < AMOUNT_TOLERANCE for found in _amounts(quote)
            )
        if isinstance(value, date):
            return value in _dates(quote, conventions.dayfirst)
        wanted, text = normalize_for_match(str(value)), normalize_for_match(quote)
        return wanted in text or fuzz.partial_ratio(wanted, text) >= STR_IN_QUOTE_THRESHOLD


class ValueComparator:
    def equal(self, a: Any, b: Any, spec: FieldSpec | None = None) -> bool:
        if a is None or b is None:
            return a is None and b is None
        na, nb = _numeric(a), _numeric(b)
        if na is not None and nb is not None:
            return abs(na - nb) <= AMOUNT_TOLERANCE
        if isinstance(a, date) or isinstance(b, date):
            return bool(a == b)
        sa, sb = normalize_for_match(str(a)), normalize_for_match(str(b))
        if spec is not None and spec.enum:
            return sa == sb
        return sa == sb or fuzz.ratio(sa, sb) >= STR_EQUAL_THRESHOLD
