"""Local OCR: text assembly in page coordinates, scan clean-up, and the real engine on every
synthetic scan profile. Engine tests need ``bhoot[ocr]`` and are marked ``ocr``."""

from __future__ import annotations

import math
from io import BytesIO

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from bhoot.adapters.ocr.assembler import OcrTextAssembler
from bhoot.adapters.ocr.rapidocr_engine import RapidOcrEngine, _split_line
from bhoot.adapters.parsing.images import DefaultPageRenderer
from bhoot.adapters.parsing.pdfium_parser import PdfiumDocumentParser
from bhoot.domain.document import BBox, Page, PageKind
from bhoot.domain.errors import ConfigError
from bhoot.engine.verification import QuoteLocator
from bhoot.ports.ocr import OcrLine, OcrPage, OcrWord, PreparedImage
from bhoot.ports.parsing import DocumentInput

LINES = [
    "Aster Bank Ltd",
    "Account Number: 718347936524",
    "IFSC: ASTR0WB00G0",
    "Opening Balance INR 9,091.72",
    "Closing Balance INR 2,07,815.11",
]


def _pdf() -> bytes:
    buf = BytesIO()
    canvas = Canvas(buf)
    canvas.setFont("Helvetica", 12)
    for n, line in enumerate(LINES):
        canvas.drawString(72, 720 - 28 * n, line)
    canvas.showPage()
    canvas.save()
    return buf.getvalue()


# ---------------------------------------------------------------- assembly (no engine needed)
def _word(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.99) -> OcrWord:
    return OcrWord(text, BBox(x0, y0, x1, y1), conf)


def test_assembly_joins_segments_on_a_baseline_and_maps_to_points() -> None:
    # two segments of one row (as OCR returns table cells), then a second row; 144 dpi = 2 px/pt
    ocr = OcrPage(
        lines=(
            OcrLine((_word("Opening", 100, 100, 240, 130), _word("Balance", 260, 102, 400, 131)), BBox(100, 100, 400, 131)),
            OcrLine((_word("9,091.72", 900, 101, 1060, 129, 0.71),), BBox(900, 101, 1060, 129)),
            OcrLine((_word("Total", 100, 200, 200, 230),), BBox(100, 200, 200, 230)),
        ),
        image_width=1200, image_height=1700, engine="test",
    )  # fmt: skip
    out = OcrTextAssembler().assemble(
        ocr, PreparedImage(b""), dpi=144, page_width_pt=600, page_height_pt=850
    )
    assert out.text == "Opening Balance 9,091.72\nTotal"
    assert [w.line_index for w in out.words] == [0, 0, 0, 1]
    assert out.words[2].bbox == BBox(450, 50.5, 530, 64.5)
    assert out.words[2].confidence == 0.71
    for char in out.chars:
        assert out.text[char.text_index] == char.char
    assert out.mean_confidence == pytest.approx((0.99 * 3 + 0.71) / 4)
    page = Page(
        0,
        600,
        850,
        PageKind.OCR_TEXT,
        out.text,
        chars=out.chars,
        words=out.words,
        text_source="ocr",
    )
    span = QuoteLocator().locate("Balance 9,091.72", page, threshold=90)
    assert span is not None
    assert page.bbox_for_span(span.start, span.end) == BBox(130, 50.5, 530, 65.5)


def test_assembly_applies_the_deskew_transform_and_clamps_to_the_page() -> None:
    shift = PreparedImage(b"", transform=(1.0, 0.0, 10.0, 0.0, 1.0, -20.0, 0.0, 0.0, 1.0))
    ocr = OcrPage((OcrLine((_word("x", 0, 40, 20, 60),), BBox(0, 40, 20, 60)),), 100, 100, "t")
    out = OcrTextAssembler().assemble(ocr, shift, dpi=72, page_width_pt=25, page_height_pt=100)
    assert out.words[0].bbox == BBox(10, 20, 25, 40)  # moved by the transform, clipped at 25
    assert (
        OcrTextAssembler()
        .assemble(OcrPage((), 1, 1, "t"), shift, dpi=72, page_width_pt=1, page_height_pt=1)
        .text
        == ""
    )


def test_word_boxes_are_estimated_when_the_engine_gives_none() -> None:
    words = _split_line("Total Due 1,250.00", BBox(0, 0, 180, 10), 0.9)
    assert [w.text for w in words] == ["Total", "Due", "1,250.00"]
    assert words[0].bbox.x0 == 0
    assert words[-1].bbox.x1 == pytest.approx(180)
    assert _split_line("   ", BBox(0, 0, 1, 1), 0.9) == ()


def test_unknown_profile_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="mobile or server"):
        RapidOcrEngine("huge")  # type: ignore[arg-type]


# ---------------------------------------------------------------- engine and preprocessing
@pytest.fixture(scope="module")
def engine() -> RapidOcrEngine:
    pytest.importorskip("rapidocr")
    found = RapidOcrEngine()
    assert found.available()
    assert found.engine_id.startswith("rapidocr-")
    assert found.engine_id.endswith("-mobile")
    return found


def _ocr_text(
    engine: RapidOcrEngine, png: bytes, dpi: int, mode: str = "auto"
) -> tuple[str, float, float]:
    from bhoot.adapters.ocr.preprocess import OpenCvScanPreprocessor

    prepared = OpenCvScanPreprocessor().prepare(png, mode=mode)  # type: ignore[arg-type]
    page = engine.recognize(prepared.png)
    with Image.open(BytesIO(png)) as img:
        width, height = img.size
    out = OcrTextAssembler().assemble(
        page, prepared, dpi=dpi, page_width_pt=width * 72 / dpi, page_height_pt=height * 72 / dpi
    )
    lowest = min((w.confidence for w in out.words), default=0.0)
    return out.text, out.mean_confidence, lowest


@pytest.mark.ocr
def test_engine_reads_a_clean_page_with_word_boxes(engine: RapidOcrEngine) -> None:
    png = DefaultPageRenderer().render(DocumentInput(_pdf()), 0, dpi=200)
    page = engine.recognize(png)
    words = [w for line in page.lines for w in line.words]
    assert page.word_count == len(words) > 10
    assert all(0 <= w.bbox.x0 < w.bbox.x1 <= page.image_width for w in words)
    assert page.mean_confidence > 0.95
    text = " ".join(w.text for w in words)
    for value in ("718347936524", "ASTR0WB00G0", "9,091.72", "2,07,815.11"):
        assert value in text


@pytest.mark.ocr
@pytest.mark.parametrize("profile", ["clean", "phone_photo", "fax", "stamped", "low_dpi"])
def test_every_scan_profile_is_read(engine: RapidOcrEngine, profile: str) -> None:
    scan_sim = pytest.importorskip("tools.synth.scan_sim")
    sim = scan_sim.ScanSimulator(scan_sim.PROFILES[profile], seed=7)
    scanned = sim.simulate(_pdf())
    dpi = 200
    png = DefaultPageRenderer().render(DocumentInput(scanned), 0, dpi=dpi)
    text, mean, _lowest = _ocr_text(engine, png, dpi)
    assert mean > 0.90
    for value in ("718347936524", "ASTR0WB00G0", "9,091.72", "2,07,815.11"):
        assert value in text, (profile, text)


@pytest.mark.ocr
def test_tilted_pages_are_straightened_and_boxes_land_on_the_real_text(
    engine: RapidOcrEngine,
) -> None:
    from bhoot.adapters.ocr.preprocess import OpenCvScanPreprocessor

    dpi = 200
    pdf = _pdf()
    truth = PdfiumDocumentParser().parse(DocumentInput(pdf), max_pages=None).pages[0]
    png = DefaultPageRenderer().render(DocumentInput(pdf), 0, dpi=dpi)
    with Image.open(BytesIO(png)) as img:
        tilted = img.convert("L").rotate(2.5, resample=Image.Resampling.BICUBIC, fillcolor=255)
        buf = BytesIO()
        tilted.save(buf, "PNG")
    pre = OpenCvScanPreprocessor()
    import cv2
    import numpy as np

    gray = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert math.isclose(abs(pre.estimate_skew(gray)), 2.5, abs_tol=0.3)
    prepared = pre.prepare(buf.getvalue(), mode="auto")
    assert any(step.startswith("deskew") for step in prepared.steps)
    out = OcrTextAssembler().assemble(
        engine.recognize(prepared.png),
        prepared,
        dpi=dpi,
        page_width_pt=truth.width,
        page_height_pt=truth.height,
    )
    ocr_page = Page(
        0, truth.width, truth.height, PageKind.OCR_TEXT, out.text, chars=out.chars, words=out.words
    )
    quote = "718347936524"
    span = QuoteLocator().locate(quote, ocr_page, threshold=90)
    assert span is not None
    found = ocr_page.bbox_for_span(span.start, span.end)
    start = truth.text.index(quote)
    expected = truth.bbox_for_span(start, start + len(quote))
    assert found is not None
    assert expected is not None
    # the page was rotated around its centre, so allow a few points either way
    assert abs(found.center[0] - expected.center[0]) < 12
    assert abs(found.center[1] - expected.center[1]) < 12


@pytest.mark.ocr
def test_preprocessing_modes(engine: RapidOcrEngine) -> None:
    from bhoot.adapters.ocr.preprocess import OpenCvScanPreprocessor

    png = DefaultPageRenderer().render(DocumentInput(_pdf()), 0, dpi=100)
    pre = OpenCvScanPreprocessor()
    assert pre.prepare(png, mode="off").png == png
    binary = pre.prepare(png, mode="aggressive")
    assert "binarize" in binary.steps
    with Image.open(BytesIO(binary.png)) as img:
        assert img.getextrema() == (0, 255)
    assert len(pre.params_hash()) == 12
