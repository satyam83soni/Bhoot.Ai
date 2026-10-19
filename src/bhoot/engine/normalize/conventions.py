"""Per-document conventions, decided once from the evidence in the whole document.

A single "03/04/2025" can't tell day-first from month-first, but a document that also contains
"25/04/2025" can. The same holds for Indian vs international digit grouping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from bhoot.domain.locales import Locale
from bhoot.engine.normalize.dates import numeric_parts
from bhoot.engine.normalize.numbers import AmountParseError, parse_amount


@dataclass(frozen=True, slots=True)
class DocConventions:
    dayfirst: bool
    # "auto": no amount told the styles apart, so each value is read with whichever valid
    # grouping it has (forcing the locale's style would reject a correct "117,861.77")
    number_grouping: Literal["indian", "international", "auto"]
    date_evidence: int = 0  # how many dates decided the order
    number_evidence: int = 0


def _grouping_vote(raw: str) -> Literal["indian", "international"] | None:
    """The one style that can read ``raw``, or None when both (or neither) can."""
    styles: tuple[Literal["indian", "international"], ...] = ("indian", "international")
    readable = []
    for style in styles:
        try:
            parse_amount(raw, style)
        except AmountParseError:
            continue
        readable.append(style)
    return readable[0] if len(readable) == 1 else None


def detect_conventions(
    dates: Iterable[str], amounts: Iterable[str], locale: Locale
) -> DocConventions:
    dayfirst_votes = monthfirst_votes = 0
    for raw in dates:
        parts = numeric_parts(raw)
        if parts is None:
            continue
        first, second = parts
        if first > 12 >= second:
            dayfirst_votes += 1
        elif second > 12 >= first:
            monthfirst_votes += 1
    if dayfirst_votes == monthfirst_votes:
        dayfirst = locale.dayfirst
    else:
        dayfirst = dayfirst_votes > monthfirst_votes

    indian = international = 0
    for raw in amounts:
        vote = _grouping_vote(str(raw))
        indian += vote == "indian"
        international += vote == "international"
    grouping: Literal["indian", "international", "auto"]
    if indian == international:
        grouping = locale.number_grouping if indian == 0 else "auto"
    else:
        grouping = "indian" if indian > international else "international"

    return DocConventions(
        dayfirst=dayfirst,
        number_grouping=grouping,
        date_evidence=dayfirst_votes + monthfirst_votes,
        number_evidence=indian + international,
    )
