"""Find a model's quote on the page: exact, then normalised, then fuzzy, then one page either
side (models are often off by one page). Spans always point into the original page text, so
they map to character boxes and a bounding box."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz

from bhoot.domain.document import Document, Page, PageKind
from bhoot.engine.normalize.text import normalize_with_index_map

MIN_FUZZY_CHARS = 4
TEXT_KINDS = frozenset({PageKind.TEXT, PageKind.OCR_TEXT, PageKind.UNCLASSIFIED})


@dataclass(frozen=True, slots=True)
class LocatedSpan:
    page: int
    start: int
    end: int
    score: float  # 100 exact, 99 normalised, else the fuzzy score

    def text(self, doc: Document) -> str:
        return doc.page(self.page).text[self.start : self.end]


def has_text(page: Page) -> bool:
    return page.kind in TEXT_KINDS and bool(page.text.strip())


class QuoteLocator:
    def locate(self, quote: str, page: Page, *, threshold: float) -> LocatedSpan | None:
        text = page.text
        if not quote or not text:
            return None
        start = text.find(quote)
        if start >= 0:
            return LocatedSpan(page.index, start, start + len(quote), 100.0)
        norm_quote, _ = normalize_with_index_map(quote)
        norm_text, back = normalize_with_index_map(text)
        if not norm_quote or not norm_text:
            return None
        start = norm_text.find(norm_quote)
        if start >= 0:
            end = start + len(norm_quote) - 1
            return LocatedSpan(page.index, back[start], back[end] + 1, 99.0)
        if len(norm_quote) < MIN_FUZZY_CHARS:
            return None
        found = fuzz.partial_ratio_alignment(norm_quote, norm_text, score_cutoff=threshold)
        if found is None or found.dest_end <= found.dest_start or found.score < threshold:
            return None
        return LocatedSpan(
            page.index, back[found.dest_start], back[found.dest_end - 1] + 1, found.score
        )

    def locate_in(
        self, quote: str, doc: Document, page_hint: int | None, *, threshold: float
    ) -> LocatedSpan | None:
        """Try the given page, then its neighbours; without a hint, every text page."""
        for index in _candidates(doc, page_hint):
            page = doc.page(index)
            if not has_text(page):
                continue
            span = self.locate(quote, page, threshold=threshold)
            if span is not None:
                return span
        return None


def _candidates(doc: Document, hint: int | None) -> Iterable[int]:
    if hint is None or not 0 <= hint < doc.page_count:
        return range(doc.page_count)
    return [i for i in (hint, hint - 1, hint + 1) if 0 <= i < doc.page_count]
