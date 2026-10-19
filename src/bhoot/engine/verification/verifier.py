"""Evidence verification: VALIDATED fields become VERIFIED only when grounded.

* Tier 0 values carry their own evidence (read straight from the text or a table cell).
* Pages with text (digital or local OCR): the quote must be found on the page and the value
  must appear inside what was found. The evidence records the real page text, span, box and,
  for OCR, the weakest word confidence in the span.
* Pages with no text (scanned, OCR unavailable): the value must at least appear in the quote
  the model gave; the evidence is marked ungrounded so confidence stays capped.
"""

from __future__ import annotations

from collections.abc import Mapping

from bhoot.domain.document import BBox, Document, Page
from bhoot.domain.evidence import Evidence, EvidenceSource
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.engine.normalize.conventions import DocConventions
from bhoot.engine.verification.locator import LocatedSpan, QuoteLocator, has_text
from bhoot.engine.verification.matching import ValueInQuoteMatcher

CODE_SOURCES = frozenset({EvidenceSource.CODE, EvidenceSource.OCR_CODE})


class EvidenceVerifier:
    def __init__(
        self, locator: QuoteLocator | None = None, matcher: ValueInQuoteMatcher | None = None
    ) -> None:
        self.locator = locator or QuoteLocator()
        self.matcher = matcher or ValueInQuoteMatcher()

    def verify(
        self,
        field_states: Mapping[str, FieldState],
        doc: Document,
        conventions: DocConventions,
        *,
        threshold: float,
    ) -> dict[str, FieldState]:
        out = dict(field_states)
        for path, fs in field_states.items():
            if fs.status is FieldStatus.VALIDATED:
                out[path] = self._verify(fs, doc, conventions, threshold)
        return out

    def _verify(
        self, fs: FieldState, doc: Document, conventions: DocConventions, threshold: float
    ) -> FieldState:
        cand = fs.current
        assert cand is not None  # VALIDATED always has a value
        if cand.is_code or (cand.evidence is not None and cand.evidence.source in CODE_SOURCES):
            return fs.transition(FieldStatus.VERIFIED, "read directly from the document")
        if not cand.quote:
            return fs.transition(FieldStatus.UNGROUNDED, "no quote was given")
        span = self.locator.locate_in(cand.quote, doc, cand.page, threshold=threshold)
        if span is not None:
            found = span.text(doc)
            if not self.matcher.matches(cand.value, found, fs.spec, conventions):
                return fs.transition(
                    FieldStatus.UNGROUNDED,
                    f"the value {cand.value} is not in the text quoted on page {span.page + 1}",
                )
            evidence = _located(span, found, doc.page(span.page))
            return fs.transition(
                FieldStatus.VERIFIED,
                "quote found on the page",
                current=cand.with_evidence(evidence),
            )
        page = (
            doc.page(cand.page)
            if cand.page is not None and 0 <= cand.page < doc.page_count
            else None
        )
        if page is not None and not has_text(page):
            if self.matcher.matches(cand.value, cand.quote, fs.spec, conventions):
                evidence = Evidence(page.index, cand.quote, EvidenceSource.LLM, match_score=0.0)
                return fs.transition(
                    FieldStatus.VERIFIED,
                    "page has no text layer: value checked against the quote only",
                    current=cand.with_evidence(evidence),
                )
            return fs.transition(FieldStatus.UNGROUNDED, "the value is not in the quote given")
        where = f"page {cand.page + 1}" if cand.page is not None else "any page"
        return fs.transition(FieldStatus.UNGROUNDED, f"the quote was not found on {where}")


def _located(span: LocatedSpan, found: str, page: Page) -> Evidence:
    bbox = page.bbox_for_span(span.start, span.end)
    return Evidence(
        page=span.page,
        quote=found,
        source=EvidenceSource.LLM,
        char_span=(span.start, span.end),
        bbox=bbox,
        match_score=span.score,
        ocr_min_confidence=_ocr_min_confidence(page, bbox),
    )


def _ocr_min_confidence(page: Page, bbox: BBox | None) -> float | None:
    if page.text_source != "ocr" or bbox is None:
        return None
    inside = [
        w.confidence
        for w in page.words
        if w.bbox.x0 < bbox.x1
        and w.bbox.x1 > bbox.x0
        and w.bbox.y0 < bbox.y1
        and w.bbox.y1 > bbox.y0
    ]
    return min(inside) if inside else None
