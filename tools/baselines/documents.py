"""Load an eval document as LLM input: page text when there is a text layer, else page images."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium

MIN_TEXT_CHARS = 40
IMAGE_SCALE = 150 / 72  # 150 DPI is enough for a vision model and keeps requests small


@dataclass(frozen=True)
class DocumentInput:
    pages_text: tuple[str, ...]  # "" for pages without usable text
    pages_png_b64: tuple[str, ...]  # rendered only for pages without usable text

    @property
    def uses_images(self) -> bool:
        return bool(self.pages_png_b64)


def load_document(path: Path) -> DocumentInput:
    doc = pdfium.PdfDocument(path)
    try:
        texts: list[str] = []
        images: list[str] = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_textpage().get_text_range().strip()
            if len(text) >= MIN_TEXT_CHARS:
                texts.append(text)
                continue
            texts.append("")
            buf = BytesIO()
            page.render(scale=IMAGE_SCALE, grayscale=True).to_pil().save(buf, "PNG", optimize=True)
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return DocumentInput(tuple(texts), tuple(images))
    finally:
        doc.close()
