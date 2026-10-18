"""Local OCR for scanned pages: preprocessing, recognition and text assembly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from bhoot.domain.document import BBox, CharBox, Word


IDENTITY_3X3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # row-major


@dataclass(frozen=True, slots=True)
class OcrWord:
    text: str
    bbox: BBox  # page image pixels
    confidence: float  # 0..1


@dataclass(frozen=True, slots=True)
class OcrLine:
    words: tuple[OcrWord, ...]
    bbox: BBox


@dataclass(frozen=True, slots=True)
class OcrPage:
    lines: tuple[OcrLine, ...]  # reading order
    image_width: int
    image_height: int
    engine: str  # e.g. "rapidocr-<version>-mobile"
    rotation_applied: int = 0

    @property
    def mean_confidence(self) -> float:
        words = [w for line in self.lines for w in line.words]
        return sum(w.confidence for w in words) / len(words) if words else 0.0

    @property
    def word_count(self) -> int:
        return sum(len(line.words) for line in self.lines)


@dataclass(frozen=True, slots=True)
class PreparedImage:
    png: bytes = field(repr=False)
    rotation: int = 0
    transform: tuple[float, ...] = IDENTITY_3X3  # prepared pixels -> original pixels
    steps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssembledText:
    text: str
    words: tuple[Word, ...]
    chars: tuple[CharBox, ...]
    mean_confidence: float


@runtime_checkable
class OcrEngine(Protocol):
    name: str
    version: str

    def available(self) -> bool: ...

    def recognize(self, png: bytes, *, lang: Sequence[str] = ("en",)) -> OcrPage: ...


@runtime_checkable
class ScanPreprocessor(Protocol):
    def prepare(
        self, png: bytes, *, mode: Literal["auto", "off", "aggressive"]
    ) -> PreparedImage: ...

    def params_hash(self) -> str: ...


@runtime_checkable
class OcrTextAssembler(Protocol):
    def assemble(
        self,
        ocr: OcrPage,
        prepared: PreparedImage,
        *,
        dpi: int,
        page_width_pt: float,
        page_height_pt: float,
    ) -> AssembledText: ...
