"""Tables for Tier-0 extractors: ruled tables read from the PDF, loaded lazily per page.

Borderless tables (no ruling lines) are rebuilt from word positions by the word-table builder,
added together with the first pack that needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bhoot.domain.document import Table
from bhoot.ports.parsing import DocumentInput, TableExtractor

HeaderMatcher = Callable[[Sequence[str]], bool]


class EngineTableProvider:
    def __init__(self, extractor: TableExtractor | None, inp: DocumentInput) -> None:
        self.extractor = extractor
        self.inp = inp
        self._cache: dict[int, tuple[Table, ...]] = {}

    def tables(
        self, page_index: int, *, header_matcher: HeaderMatcher | None = None
    ) -> tuple[Table, ...]:
        if self.extractor is None:
            return ()
        if page_index not in self._cache:
            found = self.extractor.extract_tables(self.inp, [page_index])
            self._cache[page_index] = tuple(found.get(page_index, ()))
        tables = self._cache[page_index]
        if header_matcher is None:
            return tables
        return tuple(t for t in tables if header_matcher([c.text for c in t.header_row()]))
