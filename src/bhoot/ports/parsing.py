"""Reading documents: parsers, table extraction, rendering and cropping.

All parsing ports are synchronous and CPU-bound; nodes call them in a worker thread.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from bhoot.domain.document import BBox, Document, Table, Word


@dataclass(frozen=True)
class DocumentInput:
    data: bytes = field(repr=False)
    filename: str | None = None
    password: str | None = field(default=None, repr=False)


@runtime_checkable
class DocumentParser(Protocol):
    name: str
    priority: int  # higher wins when several parsers support the same mime type

    def supports(self, mime: str) -> bool: ...

    def parse(self, inp: DocumentInput, *, max_pages: int | None) -> Document: ...


@runtime_checkable
class ParserSelector(Protocol):
    def sniff_mime(self, head: bytes) -> str: ...

    def for_input(self, inp: DocumentInput) -> DocumentParser: ...


@runtime_checkable
class TableExtractor(Protocol):
    def extract_tables(
        self, inp: DocumentInput, page_indices: Sequence[int]
    ) -> Mapping[int, tuple[Table, ...]]: ...

    def extract_words(
        self, inp: DocumentInput, page_indices: Sequence[int]
    ) -> Mapping[int, tuple[Word, ...]]: ...


@runtime_checkable
class PageRenderer(Protocol):
    def render(self, inp: DocumentInput, page_index: int, *, dpi: int) -> bytes: ...  # PNG


@runtime_checkable
class ImagePreprocessor(Protocol):
    def prepare(self, png: bytes, *, max_side_px: int, grayscale: bool) -> bytes: ...


@runtime_checkable
class CropRenderer(Protocol):
    def crop(self, png: bytes, bbox_px: BBox, *, pad_px: int, full_width: bool) -> bytes: ...
