"""Images (PNG, JPEG, multi-page TIFF, phone photos) as documents with one page per frame.

Images have no text layer: every page starts UNCLASSIFIED with empty text and goes down the
OCR / vision path. Page sizes are converted to PDF points from the image's DPI.
"""

from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image, ImageSequence, UnidentifiedImageError

from bhoot.domain.document import Document, Page, PageKind
from bhoot.domain.errors import TooManyPagesError, UnsupportedFormatError
from bhoot.ports.parsing import DocumentInput

IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/tiff"})
DEFAULT_DPI = 200.0


def image_dpi(img: Image.Image) -> float:
    dpi = img.info.get("dpi")
    if isinstance(dpi, tuple) and dpi and float(dpi[0]) > 30:
        return float(dpi[0])
    return DEFAULT_DPI


class ImageDocumentParser:
    name = "image"
    priority = 10

    def supports(self, mime: str) -> bool:
        return mime in IMAGE_MIMES

    def parse(self, inp: DocumentInput, *, max_pages: int | None) -> Document:
        try:
            img = Image.open(BytesIO(inp.data))
        except UnidentifiedImageError as exc:
            raise UnsupportedFormatError(
                f"{inp.filename or 'file'} is not a readable image"
            ) from exc
        pages: list[Page] = []
        with img:
            mime = Image.MIME.get(img.format or "", "image/png")
            for index, frame in enumerate(ImageSequence.Iterator(img)):
                if max_pages is not None and index >= max_pages:
                    raise TooManyPagesError(f"image has more than {max_pages} pages")
                dpi = image_dpi(frame)
                pages.append(
                    Page(
                        index=index,
                        width=frame.width / dpi * 72,
                        height=frame.height / dpi * 72,
                        kind=PageKind.UNCLASSIFIED,
                        text_source="none",
                    )
                )
        sha = hashlib.sha256(inp.data).hexdigest()
        return Document(
            doc_id=sha[:16], sha256=sha, mime=mime, pages=tuple(pages), source_name=inp.filename
        )
