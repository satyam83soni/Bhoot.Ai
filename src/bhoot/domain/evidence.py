"""Where a value came from on the page: the proof behind every extracted field."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bhoot.domain.document import BBox


class EvidenceSource(StrEnum):
    CODE = "code"  # Tier 0: read directly from PDF text or a table cell
    OCR_CODE = "ocr_code"  # Tier 0 on OCR text: grounded, but OCR confidence applies
    LLM = "llm"  # quote supplied by a model and located on the page by the verifier
    CROP = "crop"  # value re-read from an image crop by a vision model


@dataclass(frozen=True, slots=True)
class Evidence:
    page: int  # 0-based internally
    quote: str
    source: EvidenceSource
    char_span: tuple[int, int] | None = None  # located span in Page.text
    bbox: BBox | None = None
    match_score: float = 100.0  # 100 = exact; fuzzy score otherwise
    ocr_min_confidence: float | None = None  # weakest OCR word inside the span

    @property
    def grounded(self) -> bool:
        """True when the quote was actually found on the page (not just claimed)."""
        return self.char_span is not None or self.source in (
            EvidenceSource.CODE,
            EvidenceSource.OCR_CODE,
        )
