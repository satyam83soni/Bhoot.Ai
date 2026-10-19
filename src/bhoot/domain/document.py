"""Parsed documents: pages with text, character boxes, words and tables.

Coordinates are PDF points with the origin at the **top-left** (parsers normalise this), so
OCR pixel boxes and PDF boxes live in the same space. Pages carry an ``image_ref`` (a blob
store key), never image bytes, which keeps graph state and checkpoints small.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal


class PageKind(StrEnum):
    UNCLASSIFIED = "unclassified"  # set by parsers; replaced by page classification
    TEXT = "text"  # usable PDF text layer
    OCR_TEXT = "ocr_text"  # scanned page whose text came from local OCR
    SCANNED = "scanned"  # no usable text and OCR unavailable or failed: vision LLM
    MIXED = "mixed"  # broken text layer or large image regions
    BLANK = "blank"


@dataclass(frozen=True, slots=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"invalid bbox {self}: x1 >= x0 and y1 >= y0 required")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    @staticmethod
    def union_all(boxes: list[BBox] | tuple[BBox, ...]) -> BBox | None:
        if not boxes:
            return None
        out = boxes[0]
        for box in boxes[1:]:
            out = out.union(box)
        return out


@dataclass(frozen=True, slots=True)
class CharBox:
    char: str
    bbox: BBox
    text_index: int  # index into Page.text, so a character span maps to a bbox


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    bbox: BBox
    confidence: float = 1.0  # 1.0 for PDF text; OCR score (0..1) for OCR words
    line_index: int | None = None


@dataclass(frozen=True, slots=True)
class TableCell:
    row: int
    col: int
    text: str
    bbox: BBox | None = None


@dataclass(frozen=True, slots=True)
class Table:
    page_index: int
    bbox: BBox | None
    cells: tuple[TableCell, ...]
    n_rows: int
    n_cols: int

    def row(self, r: int) -> tuple[TableCell, ...]:
        return tuple(sorted((c for c in self.cells if c.row == r), key=lambda c: c.col))

    def header_row(self) -> tuple[TableCell, ...]:
        return self.row(0)

    def rows(self) -> list[tuple[TableCell, ...]]:
        return [self.row(r) for r in range(self.n_rows)]


@dataclass(frozen=True, slots=True)
class CropRequest:
    """A region of a page image to re-read (crop repair on OCR pages)."""

    page: int  # 0-based
    bbox: BBox  # page coordinates (points, top-left origin)
    paths: tuple[str, ...]  # concrete paths the crop should re-read
    reason: str


TextSource = Literal["pdf", "ocr", "none"]


@dataclass(frozen=True, slots=True)
class Page:
    index: int  # 0-based internally; exposed 1-based in results
    width: float
    height: float
    kind: PageKind = PageKind.UNCLASSIFIED
    text: str = ""
    chars: tuple[CharBox, ...] = ()
    words: tuple[Word, ...] = ()
    tables: tuple[Table, ...] = ()
    image_ref: str | None = None
    text_coverage: float = 0.0
    text_source: TextSource = "pdf"
    ocr_confidence: float | None = None
    rotation: int = 0

    @property
    def number(self) -> int:
        """1-based page number, as shown to users."""
        return self.index + 1

    def bbox_for_span(self, start: int, end: int) -> BBox | None:
        """Union of character boxes for ``text[start:end]`` (None if no boxes cover it)."""
        boxes = [c.bbox for c in self.chars if start <= c.text_index < end and not c.char.isspace()]
        return BBox.union_all(boxes)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    sha256: str
    mime: str
    pages: tuple[Page, ...]
    source_name: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, index: int) -> Page:
        return self.pages[index]

    def with_pages(self, pages: tuple[Page, ...]) -> Document:
        return replace(self, pages=pages)

    def replace_page(self, page: Page) -> Document:
        pages = list(self.pages)
        pages[page.index] = page
        return replace(self, pages=tuple(pages))

    def pages_of_kind(self, *kinds: PageKind) -> tuple[Page, ...]:
        return tuple(p for p in self.pages if p.kind in kinds)
