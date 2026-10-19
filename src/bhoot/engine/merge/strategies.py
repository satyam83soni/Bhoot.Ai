"""Merge strategies: how answers from several calls (pages, chunks, rounds) become one value
per field.

* ``scalar_prefer_verified``: one value per path. Code beats a model; an answer with a quote
  beats one without; the earliest page wins ties.
* ``rows_by_page``: table rows from every page in page order, with repeated carried-forward
  rows and pseudo-rows (opening/closing balance, totals, B/F, C/F) removed.
* ``list_concat``: other lists, in order, exact duplicates removed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from bhoot.domain.candidate import CandidateValue
from bhoot.domain.schema import FieldPath
from bhoot.engine.registry import StrategyRegistry

PSEUDO_ROW = re.compile(
    r"^\s*(opening\s+balance|closing\s+balance|balance\s+(b/?f|c/?f|brought|carried)|"
    r"(b/?f|c/?f)\b|brought\s+forward|carried\s+forward|total|grand\s+total)",
    re.IGNORECASE,
)

Row = dict[str, CandidateValue]  # column (relative path) -> candidate


@dataclass(frozen=True)
class ItemOrder:
    """Where an answer came from, so rows can be put in document order."""

    first_page: int
    position: int  # order of the work item in the round


class ScalarStrategy(Protocol):
    def pick(
        self, candidates: Sequence[CandidateValue], order: Mapping[str, ItemOrder]
    ) -> CandidateValue: ...


class ListStrategy(Protocol):
    def combine(
        self, items: Sequence[tuple[ItemOrder, list[Row]]]
    ) -> tuple[list[Row], list[str]]: ...  # rows in final order, warnings


class ScalarPreferVerified:
    def pick(
        self, candidates: Sequence[CandidateValue], order: Mapping[str, ItemOrder]
    ) -> CandidateValue:
        def rank(c: CandidateValue) -> tuple[int, int, int, int]:
            where = order.get(c.work_item_id, ItemOrder(10**6, 10**6))
            page = c.page if c.page is not None else where.first_page
            return (0 if c.is_code else 1, 0 if c.quote else 1, page, where.position)

        return min(candidates, key=rank)


def _row_key(row: Row) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((col, "" if c.raw is None else str(c.raw)) for col, c in row.items()))


def _is_pseudo(row: Row) -> bool:
    texts = [str(c.raw) for c in row.values() if isinstance(c.raw, str)]
    return any(
        PSEUDO_ROW.match(t)
        for t in texts
        if not t.replace(",", "").replace(".", "").strip().isdigit()
    )


class RowsByPage:
    def combine(self, items: Sequence[tuple[ItemOrder, list[Row]]]) -> tuple[list[Row], list[str]]:
        rows: list[Row] = []
        warnings: list[str] = []
        for _, item_rows in sorted(items, key=lambda pair: (pair[0].first_page, pair[0].position)):
            for k, row in enumerate(item_rows):
                if _is_pseudo(row):
                    continue
                if k == 0 and rows and _row_key(rows[-1]) == _row_key(row):
                    warnings.append("dropped a row repeated at the top of the next page")
                    continue
                rows.append(row)
        return rows, warnings


class ListConcat:
    def combine(self, items: Sequence[tuple[ItemOrder, list[Row]]]) -> tuple[list[Row], list[str]]:
        rows: list[Row] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for _, item_rows in sorted(items, key=lambda pair: (pair[0].first_page, pair[0].position)):
            for row in item_rows:
                key = _row_key(row)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        return rows, []


SCALAR_STRATEGIES: StrategyRegistry[ScalarStrategy] = StrategyRegistry(
    "merge strategy", {"scalar_prefer_verified": ScalarPreferVerified()}
)
LIST_STRATEGIES: StrategyRegistry[ListStrategy] = StrategyRegistry(
    "merge strategy", {"rows_by_page": RowsByPage(), "list_concat": ListConcat()}
)


def relative(path: FieldPath, array: FieldPath) -> str:
    """``transactions[3].debit`` relative to ``transactions`` -> ``debit`` ("" for scalar lists)."""
    parts = path.parts()[len(array.parts()) + 1 :]
    return FieldPath.from_parts(parts).raw if parts else ""
