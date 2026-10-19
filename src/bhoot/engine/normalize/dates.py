"""Read dates as printed: "03/04/2025", "03-Apr-2025", "3 April 2025", "2025-04-03", "03.04.25".

Day-first vs month-first is decided per document (see conventions); a single date like
"03/04/2025" is genuinely ambiguous on its own, and ``is_ambiguous`` says so.
"""

from __future__ import annotations

import re
from datetime import date, datetime

EMPTY_TOKENS = frozenset({"", "-", "--", "nil", "n/a", "na", "null", "none"})

_DAYFIRST = (
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
)  # fmt: skip
_MONTHFIRST = (
    "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",
)  # fmt: skip
_UNAMBIGUOUS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y", "%d-%b-%y", "%d %b %y", "%d %B %Y",
    "%d-%B-%Y", "%b %d, %Y", "%B %d, %Y", "%d%b%Y", "%d %b, %Y",
)  # fmt: skip
_NUMERIC = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})$")


class DateParseError(ValueError):
    """The text is not a readable date."""


def _clean(raw: str) -> str:
    text = " ".join(raw.strip().split())
    return re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", text, flags=re.IGNORECASE)  # "3rd April 2025"


def parse_date(raw: str | date | None, *, dayfirst: bool = True) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    text = _clean(raw)
    if text.casefold() in EMPTY_TOKENS:
        return None
    numeric = _DAYFIRST + _MONTHFIRST if dayfirst else _MONTHFIRST + _DAYFIRST
    for fmt in (*_UNAMBIGUOUS, *numeric):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise DateParseError(f"{raw!r} is not a recognised date")


def numeric_parts(raw: str) -> tuple[int, int] | None:
    """(first, second) number of a numeric date like 03/04/2025, else None."""
    m = _NUMERIC.match(_clean(raw))
    return (int(m.group(1)), int(m.group(2))) if m else None


def is_ambiguous(raw: str) -> bool:
    """True for numeric dates where both readings are valid and different (03/04 vs 04/03)."""
    parts = numeric_parts(raw)
    return parts is not None and parts[0] <= 12 and parts[1] <= 12 and parts[0] != parts[1]
