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
    number_grouping: Literal["indian", "international"]
    date_evidence: int = 0  # how many dates decided the order
    number_evidence: int = 0


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
        if str(raw).count(",") < 2:  # one comma can't tell the two styles apart
            continue
        try:
            found = parse_amount(raw).grouping
        except AmountParseError:
            continue
        indian += found == "indian"
        international += found == "international"
    if indian == international:
        grouping: Literal["indian", "international"] = (
            "international" if locale.number_grouping == "international" else "indian"
        )
    else:
        grouping = "indian" if indian > international else "international"

    return DocConventions(
        dayfirst=dayfirst,
        number_grouping=grouping,
        date_evidence=dayfirst_votes + monthfirst_votes,
        number_evidence=indian + international,
    )
