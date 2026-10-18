"""Read amounts the way they are printed: "₹25,00,000.00", "1,234.00 Dr", "(500)", "Rs. 500/-".

The parser refuses instead of guessing: a digit grouping that is neither Indian nor
international (often an OCR or model slip) or a European-style "1.234,56" raises, so the field
fails a type check and gets repaired rather than silently becoming a wrong number.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from bhoot.domain.values import Grouping, ParsedAmount, Side

GroupingPolicy = Literal["indian", "international", "auto"]

EMPTY_TOKENS = frozenset({"", "-", "--", "—", "nil", "n/a", "na", "null", "none"})

_CURRENCY = re.compile(r"(?i)(₹|\binr\b|\brs\.?|\brupees\b)")
_SIDE = re.compile(r"(?i)(?<![a-z])(dr|cr)\.?(?![a-z])")  # also "1,234.00CR"
_SPACE_AROUND_SEP = re.compile(r"\s*([,.])\s*")
_INDIAN = re.compile(r"^\d{1,2}(,\d{2})*,\d{3}(\.\d+)?$")
_INTERNATIONAL = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_PLAIN = re.compile(r"^\d+(\.\d+)?$|^\.\d+$")


class AmountParseError(ValueError):
    """The text is not a readable amount."""


class AmbiguousNumberError(AmountParseError):
    """The text could mean two different numbers (e.g. '1.234,56')."""


class MalformedNumberError(AmountParseError):
    """Digit grouping or characters do not form a valid number."""


def parse_amount(
    raw: str | int | float | Decimal | None, grouping: GroupingPolicy = "auto"
) -> ParsedAmount:
    if raw is None:
        return ParsedAmount(None, "")
    if isinstance(raw, (int, Decimal)):
        return ParsedAmount(Decimal(raw), str(raw))
    if isinstance(raw, float):
        return ParsedAmount(Decimal(str(raw)), str(raw))

    original = raw
    text = raw.strip()
    if text.casefold() in EMPTY_TOKENS:
        return ParsedAmount(None, original)

    text = _CURRENCY.sub("", text)
    text = text.replace("/-", "").strip()

    side: Side | None = None
    side_match = _SIDE.search(text)
    if side_match:
        side = "debit" if side_match.group(1).casefold() == "dr" else "credit"
        text = _SIDE.sub("", text).strip()

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1].strip()
    if text.startswith("-"):
        negative, text = not negative, text[1:].strip()
    elif text.endswith("-"):  # "1,234.00-" as printed by some banks
        negative, text = True, text[:-1].strip()
    if text.startswith("+"):
        text = text[1:].strip()

    text = _SPACE_AROUND_SEP.sub(r"\1", text)  # OCR: "1, 234 .00" -> "1,234.00"
    if text.casefold() in EMPTY_TOKENS:
        return ParsedAmount(None, original, side)

    if "," in text and "." in text and text.rfind(",") > text.rfind("."):
        raise AmbiguousNumberError(
            f"{original!r} looks like European formatting (comma as decimal mark)"
        )

    found = _grouping_of(text)
    if found is None:
        raise MalformedNumberError(f"{original!r} is not a valid amount")
    if grouping != "auto" and found not in (grouping, "none") and not _fits_both(text):
        raise MalformedNumberError(f"{original!r} does not use {grouping} digit grouping")

    try:
        value = Decimal(text.replace(",", ""))
    except InvalidOperation as exc:  # pragma: no cover - patterns above already guarantee digits
        raise MalformedNumberError(f"{original!r} is not a valid amount") from exc
    if negative or side == "debit":
        value = -value
    return ParsedAmount(value, original, side, found)


def _grouping_of(text: str) -> Grouping | None:
    if _PLAIN.fullmatch(text):
        return "none"
    indian, international = bool(_INDIAN.fullmatch(text)), bool(_INTERNATIONAL.fullmatch(text))
    if indian and international:  # one comma ("12,345.00") is valid in both styles
        return "international"
    if indian:
        return "indian"
    if international:
        return "international"
    return None


def _fits_both(text: str) -> bool:
    return bool(_INDIAN.fullmatch(text)) and bool(_INTERNATIONAL.fullmatch(text))


def format_amount(value: Decimal, grouping: Literal["indian", "international"] = "indian") -> str:
    """Inverse of parse_amount for positive values (used by tests and prompts)."""
    sign = "-" if value < 0 else ""
    integer, _, frac = f"{abs(value):f}".partition(".")
    if grouping == "international":
        grouped = f"{int(integer):,}"
    else:
        head, tail = integer[:-3], integer[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        grouped = ",".join([*groups, tail])
    return f"{sign}{grouped}" + (f".{frac}" if frac else "")
