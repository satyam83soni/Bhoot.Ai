"""Parsing adapters against the synthetic eval documents and hand-made edge cases."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from bhoot.adapters.parsing.factory import MimeParserSelector, sniff_mime
from bhoot.adapters.parsing.image_parser import ImageDocumentParser
from bhoot.adapters.parsing.images import (
    DefaultPageRenderer,
    PillowCropRenderer,
    PillowPreprocessor,
)
from bhoot.adapters.parsing.pdfium_parser import PdfiumDocumentParser
from bhoot.adapters.parsing.plumber_tables import PlumberTableExtractor
from bhoot.domain.document import BBox, PageKind
from bhoot.domain.errors import (
    CorruptPdfError,
    EncryptedPdfError,
    TooManyPagesError,
    UnsupportedFormatError,
)
from bhoot.ports import DocumentParser, PageRenderer, ParserSelector, TableExtractor
from bhoot.ports.parsing import DocumentInput

EVALS = Path(__file__).resolve().parents[3] / "evals"


def _case(dataset: str, case: str) -> tuple[DocumentInput, dict]:  # type: ignore[type-arg]
    folder = EVALS / dataset / case
    data = (folder / "document.pdf").read_bytes()
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    expected["_layout"] = meta.get("layout") or meta.get("renderer")
    return DocumentInput(data, filename=f"{case}.pdf"), expected


def _pdf(pages: int = 1, password: str | None = None) -> bytes:
    buf = BytesIO()
    canvas = Canvas(buf, encrypt=password) if password else Canvas(buf)
    for i in range(pages):
        canvas.drawString(72, 720, f"Invoice total 1,250.00 on page {i + 1}")
        canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _png(size: tuple[int, int] = (400, 200), dpi: int = 200) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG", dpi=(dpi, dpi))
    return buf.getvalue()


# ---------------------------------------------------------------- selector
def test_sniff_mime() -> None:
    assert sniff_mime(b"%PDF-1.7 ...") == "application/pdf"
    assert sniff_mime(b"junk\n%PDF-1.4") == "application/pdf"
    assert sniff_mime(_png()[:16]) == "image/png"
    assert sniff_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_mime(b"II*\x00rest") == "image/tiff"
    assert sniff_mime(b"PK\x03\x04") == "application/octet-stream"


def test_selector_picks_parser_and_rejects_unknown_formats() -> None:
    selector = MimeParserSelector([ImageDocumentParser(), PdfiumDocumentParser()])
    assert isinstance(selector, ParserSelector)
    assert selector.for_input(DocumentInput(_pdf())).name == "pdfium"
    assert selector.for_input(DocumentInput(_png())).name == "image"
    with pytest.raises(UnsupportedFormatError, match="not supported"):
        selector.for_input(DocumentInput(b"PK\x03\x04 zip file", filename="x.docx"))


# ---------------------------------------------------------------- PDF parsing
def test_pdf_text_and_char_boxes_line_up() -> None:
    inp, expected = _case("bank_statement", "case_0001")
    parser = PdfiumDocumentParser()
    assert isinstance(parser, DocumentParser)
    doc = parser.parse(inp, max_pages=None)
    first = doc.page(0)
    assert first.kind is PageKind.UNCLASSIFIED
    assert expected["account_number"] in first.text
    for box in first.chars[:200]:
        assert first.text[box.text_index] == box.char
        assert 0 <= box.bbox.x0 < box.bbox.x1 <= first.width
        assert 0 <= box.bbox.y0 < box.bbox.y1 <= first.height
    start = first.text.index(expected["account_number"])
    span = first.bbox_for_span(start, start + len(expected["account_number"]))
    assert span is not None
    assert span.y0 < first.height / 2  # header block sits in the top half (top-left origin)
    assert 0 < first.text_coverage < 1
    assert doc.sha256
    assert doc.mime == "application/pdf"


def test_pdf_page_limit() -> None:
    with pytest.raises(TooManyPagesError, match="limit is 2"):
        PdfiumDocumentParser().parse(DocumentInput(_pdf(pages=3)), max_pages=2)


def test_encrypted_pdf_needs_the_password() -> None:
    data = _pdf(password="s3cret")
    parser = PdfiumDocumentParser()
    with pytest.raises(EncryptedPdfError, match="password"):
        parser.parse(DocumentInput(data), max_pages=None)
    with pytest.raises(EncryptedPdfError, match="wrong password"):
        parser.parse(DocumentInput(data, password="nope"), max_pages=None)
    doc = parser.parse(DocumentInput(data, password="s3cret"), max_pages=None)
    assert "1,250.00" in doc.page(0).text


def test_corrupt_pdf() -> None:
    with pytest.raises(CorruptPdfError):
        PdfiumDocumentParser().parse(DocumentInput(b"%PDF-1.4 this is not a pdf"), max_pages=None)


def test_password_is_not_in_repr() -> None:
    assert "s3cret" not in repr(DocumentInput(b"x", password="s3cret"))


# ---------------------------------------------------------------- tables and words
def test_ruled_tables_are_found_with_header_and_cell_boxes() -> None:
    inp, _ = _case("bank_statement", "case_0001")  # bank_a: gridded layout
    extractor = PlumberTableExtractor()
    assert isinstance(extractor, TableExtractor)
    [table] = extractor.extract_tables(inp, [0])[0]
    header = [c.text for c in table.header_row()]
    assert header[0] == "Date"
    assert "Closing Balance" in header
    assert table.n_rows > 10
    assert all(c.bbox is not None for c in table.header_row())


def test_borderless_pages_still_give_positioned_words() -> None:
    inp, expected = _case("bank_statement", "case_0002")  # bank_b: borderless
    assert PlumberTableExtractor().extract_tables(inp, [0])[0] == ()
    words = PlumberTableExtractor().extract_words(inp, [0])[0]
    texts = {w.text for w in words}
    assert expected["account_number"] in texts
    assert all(w.bbox.x1 > w.bbox.x0 for w in words)


# ---------------------------------------------------------------- images
def test_image_documents() -> None:
    doc = ImageDocumentParser().parse(DocumentInput(_png((400, 200), dpi=200)), max_pages=None)
    assert doc.page_count == 1
    assert doc.mime == "image/png"
    assert doc.page(0).width == pytest.approx(400 / 200 * 72, rel=1e-3)  # PNG stores DPI rounded
    assert doc.page(0).text == ""

    buf = BytesIO()
    frames = [Image.new("L", (100, 100), v) for v in (0, 128, 255)]
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    tiff = ImageDocumentParser().parse(DocumentInput(buf.getvalue()), max_pages=None)
    assert tiff.page_count == 3
    with pytest.raises(TooManyPagesError):
        ImageDocumentParser().parse(DocumentInput(buf.getvalue()), max_pages=2)


def test_render_pdf_and_image_pages() -> None:
    renderer = DefaultPageRenderer()
    assert isinstance(renderer, PageRenderer)
    png = renderer.render(DocumentInput(_pdf()), 0, dpi=144)
    with Image.open(BytesIO(png)) as img:
        assert img.width == pytest.approx(595.28 * 2, abs=2)  # A4 page at 144 dpi = 2x
    png = renderer.render(DocumentInput(_png((400, 200), dpi=100)), 0, dpi=200)
    with Image.open(BytesIO(png)) as img:
        assert img.size == (800, 400)
    small = DefaultPageRenderer(max_side_px=500).render(DocumentInput(_pdf()), 0, dpi=300)
    with Image.open(BytesIO(small)) as img:
        assert max(img.size) <= 500


def test_preprocess_and_crop() -> None:
    page = _png((1000, 1400))
    prepared = PillowPreprocessor().prepare(page, max_side_px=700, grayscale=True)
    with Image.open(BytesIO(prepared)) as img:
        assert max(img.size) == 700
        assert img.mode == "L"
    crop = PillowCropRenderer().crop(page, BBox(100, 500, 300, 530), pad_px=8, full_width=True)
    with Image.open(BytesIO(crop)) as img:
        assert img.size == (1000, 46)
    narrow = PillowCropRenderer().crop(page, BBox(100, 500, 300, 530), pad_px=0, full_width=False)
    with Image.open(BytesIO(narrow)) as img:
        assert img.size == (200, 30)
    with pytest.raises(ValueError, match="outside"):
        PillowCropRenderer().crop(page, BBox(5000, 5000, 5100, 5100), pad_px=0, full_width=False)
