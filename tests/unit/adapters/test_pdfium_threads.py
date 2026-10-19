"""PDFium is not thread-safe: parsing and rendering from many threads at once must not crash."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from reportlab.pdfgen.canvas import Canvas

from bhoot.adapters.parsing.images import DefaultPageRenderer
from bhoot.adapters.parsing.pdfium_parser import PdfiumDocumentParser
from bhoot.ports.parsing import DocumentInput


def _pdf(n: int) -> bytes:
    buf = BytesIO()
    canvas = Canvas(buf)
    for page in range(3):
        canvas.drawString(72, 720, f"document {n} page {page + 1} total 1,250.00")
        canvas.showPage()
    canvas.save()
    return buf.getvalue()


def test_parallel_parsing_and_rendering() -> None:
    parser, renderer = PdfiumDocumentParser(), DefaultPageRenderer()

    def work(n: int) -> tuple[int, int]:
        inp = DocumentInput(_pdf(n))
        doc = parser.parse(inp, max_pages=None)
        png = renderer.render(inp, n % 3, dpi=50)
        return doc.page_count, len(png)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(40)))
    assert all(pages == 3 and size > 0 for pages, size in results)
