"""PDF parsing with pypdfium2 (Apache-2.0 / BSD): text, character boxes and page sizes.

Coordinates are converted to the domain convention (PDF points, origin top-left).
Words and tables come from pdfplumber, lazily, only for pages that need them.
"""

from __future__ import annotations

import hashlib

import pypdfium2 as pdfium

from bhoot.domain.document import BBox, CharBox, Document, Page, PageKind
from bhoot.domain.errors import CorruptPdfError, EncryptedPdfError, TooManyPagesError
from bhoot.ports.parsing import DocumentInput

PDF_MIME = "application/pdf"


def open_pdf(inp: DocumentInput) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(inp.data, password=inp.password)
    except pdfium.PdfiumError as exc:
        message = str(exc).lower()
        if "password" in message:
            hint = "wrong password" if inp.password else "the PDF is encrypted: pass password="
            raise EncryptedPdfError(f"cannot open {inp.filename or 'document'}: {hint}") from exc
        raise CorruptPdfError(f"cannot read {inp.filename or 'document'} as a PDF: {exc}") from exc


class PdfiumDocumentParser:
    name = "pdfium"
    priority = 10

    def supports(self, mime: str) -> bool:
        return mime == PDF_MIME

    def parse(self, inp: DocumentInput, *, max_pages: int | None) -> Document:
        pdf = open_pdf(inp)
        try:
            count = len(pdf)
            if max_pages is not None and count > max_pages:
                raise TooManyPagesError(
                    f"{inp.filename or 'document'} has {count} pages; the limit is {max_pages} "
                    "(raise max_pages or pass pages=...)"
                )
            pages = tuple(self._page(pdf, i) for i in range(count))
        finally:
            pdf.close()
        sha = hashlib.sha256(inp.data).hexdigest()
        return Document(
            doc_id=sha[:16], sha256=sha, mime=PDF_MIME, pages=pages, source_name=inp.filename
        )

    @staticmethod
    def _page(pdf: pdfium.PdfDocument, index: int) -> Page:
        page = pdf[index]
        try:
            width, height = page.get_size()
            textpage = page.get_textpage()
            try:
                count = textpage.count_chars()
                text = textpage.get_text_range()
                if len(text) != count:  # keep text and char indices aligned no matter what
                    text = "".join(textpage.get_text_range(k, 1) or " " for k in range(count))
                chars: list[CharBox] = []
                covered = 0.0
                for k in range(count):
                    ch = text[k]
                    if ch.isspace():
                        continue
                    left, bottom, right, top = textpage.get_charbox(k)
                    if right <= left or top <= bottom:
                        continue
                    box = BBox(left, max(0.0, height - top), right, max(0.0, height - bottom))
                    chars.append(CharBox(ch, box, k))
                    covered += box.width * box.height
            finally:
                textpage.close()
            return Page(
                index=index,
                width=width,
                height=height,
                kind=PageKind.UNCLASSIFIED,
                text=text,
                chars=tuple(chars),
                text_coverage=covered / (width * height) if width and height else 0.0,
                text_source="pdf" if text.strip() else "none",
                rotation=page.get_rotation(),
            )
        finally:
            page.close()
