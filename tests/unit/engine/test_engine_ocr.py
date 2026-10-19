"""Scanned documents through the whole engine: OCR text instead of images, grounding on OCR
words, crop re-reads for low-confidence critical values, and the vision fallback."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from decimal import Decimal
from io import BytesIO
from typing import Any

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from bhoot.adapters.llm.fake import FakeLLMClient
from bhoot.adapters.ocr import OcrTextAssembler
from bhoot.adapters.parsing.factory import MimeParserSelector
from bhoot.adapters.parsing.image_parser import ImageDocumentParser
from bhoot.adapters.parsing.images import (
    DefaultPageRenderer,
    PillowCropRenderer,
    PillowPreprocessor,
)
from bhoot.adapters.parsing.pdfium_parser import PdfiumDocumentParser
from bhoot.adapters.storage import MemoryBlobStore
from bhoot.config.models import ModelSpec, ModelTierConfig
from bhoot.config.settings import BhootSettings, OcrSettings
from bhoot.domain.document import BBox
from bhoot.domain.job import ExtractRequest
from bhoot.domain.modes import Mode
from bhoot.domain.options import ExtractOptions
from bhoot.domain.result import ExtractionResult, JobStatus
from bhoot.domain.schema_dsl import Schema, field
from bhoot.domain.schema_introspect import introspect
from bhoot.engine.job_factory import JobFactory
from bhoot.engine.runner import EngineRunner
from bhoot.engine.services import EngineServices
from bhoot.ports.llm import LLMRequest
from bhoot.ports.ocr import OcrLine, OcrPage, OcrWord, PreparedImage

LINES = ["Aster Bank Ltd", "Account Number: 718347936524", "Closing Balance INR 2,07,815.11"]


class Summary(Schema):
    account_number: str = field(criticality="critical")
    closing_balance: Decimal = field(criticality="critical")


def _pdf(lines: Sequence[str] = LINES) -> bytes:
    buf = BytesIO()
    canvas = Canvas(buf)
    canvas.setFont("Helvetica", 12)
    for n, line in enumerate(lines):
        canvas.drawString(72, 720 - 28 * n, line)
    canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _scan_png(dpi: int = 150) -> bytes:
    """A page with no text layer: the PDF rendered to an image."""
    from bhoot.ports.parsing import DocumentInput

    png = DefaultPageRenderer().render(DocumentInput(_pdf()), 0, dpi=dpi)
    with Image.open(BytesIO(png)) as img:
        out = BytesIO()
        img.save(out, "PNG", dpi=(dpi, dpi))
    return out.getvalue()


class FakeOcr:
    """Deterministic OCR: the known lines at fixed places, with chosen word confidences."""

    name = "fake"
    version = "1"

    def __init__(self, confidence: dict[str, float] | None = None, default: float = 0.99) -> None:
        self.confidence = confidence or {}
        self.default = default
        self.calls = 0

    def available(self) -> bool:
        return True

    def recognize(self, png: bytes, *, lang: Sequence[str] = ("en",)) -> OcrPage:
        self.calls += 1
        with Image.open(BytesIO(png)) as img:
            width, height = img.size
        char_w, line_h = width * 0.012, height * 0.02
        lines = []
        for n, text in enumerate(LINES):
            y0 = height * (0.1 + 0.04 * n)
            x = width * 0.1
            words = []
            for word in text.split():
                box = BBox(x, y0, x + char_w * len(word), y0 + line_h)
                words.append(OcrWord(word, box, self.confidence.get(word, self.default)))
                x = box.x1 + char_w
            lines.append(
                OcrLine(tuple(words), BBox(words[0].bbox.x0, y0, words[-1].bbox.x1, y0 + line_h))
            )
        return OcrPage(tuple(lines), width, height, "fake-1")


class PassThrough:
    def prepare(self, png: bytes, *, mode: str) -> PreparedImage:
        return PreparedImage(png)

    def params_hash(self) -> str:
        return "none"


def reader(request: LLMRequest) -> dict[str, Any]:
    """Answers from whatever page text the engine sent (OCR text here)."""
    text = "".join(p.text or "" for p in request.messages[-1].parts)
    answer: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("Account Number:"):
            answer["account_number"] = {
                "value": line.split(":")[1].strip(),
                "quote": line,
                "page": 1,
            }
        if line.startswith("Closing Balance"):
            answer["closing_balance"] = {"value": line.split()[-1], "quote": line, "page": 1}
    return answer


async def _run(
    ocr: Any, *, enabled: bool = True, preprocessor: Any = None, mode: Mode = Mode.BALANCED,
    llm: FakeLLMClient | None = None, data: bytes | None = None,
) -> tuple[ExtractionResult[Any], FakeLLMClient, MemoryBlobStore]:  # fmt: skip
    models = ModelTierConfig(
        cheap=ModelSpec(model="openai/cheap"), strong=ModelSpec(model="openai/strong")
    )
    settings = BhootSettings(models=models, ocr=OcrSettings(enabled=enabled, dpi=150))
    fake = llm or FakeLLMClient()
    if not fake.rules:
        fake.when().respond(reader)
    blobs = MemoryBlobStore()
    services = EngineServices(
        settings=settings, llm=fake, blobs=blobs,
        parsers=MimeParserSelector([PdfiumDocumentParser(), ImageDocumentParser()]),
        renderer=DefaultPageRenderer(), preprocessor=PillowPreprocessor(), crops=PillowCropRenderer(),
        ocr=ocr, scan_preprocessor=preprocessor or PassThrough(), ocr_assembler=OcrTextAssembler(),
    )  # fmt: skip
    content = data or _scan_png()
    ref = await blobs.put(content, namespace="inputs", suffix=".png")
    req = ExtractRequest(
        doc_sha256=hashlib.sha256(content).hexdigest(), input_ref=ref, schema=introspect(Summary),
        mode=mode, options=ExtractOptions(include_trace=True),
    )  # fmt: skip
    job = JobFactory(settings, ocr_engine=services.ocr_engine_id).create(req)
    return await EngineRunner(services).run(job), fake, blobs


def _images_sent(fake: FakeLLMClient, calls: int | None = None) -> int:
    chosen = fake.calls if calls is None else fake.calls[:calls]
    return sum(1 for call in chosen for m in call.messages for p in m.parts if p.type == "image")


# ---------------------------------------------------------------- OCR text instead of images
async def test_scanned_page_is_read_by_ocr_and_grounded_on_its_words() -> None:
    ocr = FakeOcr()
    result, fake, _ = await _run(ocr)
    assert result.status is JobStatus.SUCCEEDED, result.review_reasons
    assert result.document is not None
    assert result.document.page_kinds == {1: "ocr_text"}
    assert result.versions.ocr_engine == "fake-1-mobile"
    assert result.data["closing_balance"] == Decimal("207815.11")
    closing = result.fields["closing_balance"]
    assert closing.bbox is not None  # grounded on OCR word boxes
    assert closing.trace is not None
    assert closing.trace.ocr_min_confidence == pytest.approx(0.99)
    assert _images_sent(fake) == 0  # text, not a page image: far cheaper
    assert len(fake.calls) == 1
    assert ocr.calls == 1


async def test_low_confidence_critical_value_is_re_read_from_an_image_crop() -> None:
    llm = FakeLLMClient()
    llm.when(kind="crop").respond(
        lambda req: {
            "closing_balance": {
                "value": "2,07,815.11",
                "quote": "Closing Balance INR 2,07,815.11",
                "page": 1,
            }
        }
    )
    llm.when(kind="extract").respond(reader)
    result, fake, blobs = await _run(FakeOcr({"2,07,815.11": 0.6}), llm=llm)
    assert result.status is JobStatus.SUCCEEDED, result.review_reasons
    closing = result.fields["closing_balance"]
    assert closing.trace is not None
    assert "r0:rule3:repair" in closing.trace.decisions
    assert closing.confidence >= 0.97  # the crop confirmed the OCR reading
    crop_calls = [c for c in fake.calls if c.metadata["kind"] == "crop"]
    assert len(crop_calls) == 1
    images = [p for m in crop_calls[0].messages for p in m.parts if p.type == "image"]
    assert len(images) == 1
    assert images[0].image_ref is not None
    assert images[0].image_ref.startswith("crops/")
    assert await blobs.exists(images[0].image_ref)
    with Image.open(BytesIO(await blobs.get(images[0].image_ref))) as crop:
        assert crop.height < crop.width / 5  # a thin strip around the row, not the page
    account = result.fields["account_number"]
    assert account.trace is not None
    assert "rule3" not in " ".join(account.trace.decisions)  # confident values are not re-read


# ---------------------------------------------------------------- vision fallback
async def test_unreadable_scan_falls_back_to_the_vision_model() -> None:
    result, fake, _ = await _run(FakeOcr(default=0.5))
    assert result.document is not None
    assert result.document.page_kinds == {1: "scanned"}
    assert _images_sent(fake, calls=1) == 1  # the first extraction sees the page image
    assert any("OCR confidence too low on page(s) [1]" in w for w in result.warnings)


@pytest.mark.parametrize(
    ("ocr", "enabled", "hint"), [(None, True, "install bhoot[ocr]"), (FakeOcr(), False, "disabled")]
)
async def test_missing_or_disabled_ocr_uses_vision_and_says_why(
    ocr: Any, enabled: bool, hint: str
) -> None:
    result, fake, _ = await _run(ocr, enabled=enabled)
    assert _images_sent(fake, calls=1) == 1
    assert any(hint in w for w in result.warnings)
    assert result.versions.ocr_engine is None


async def test_digital_pages_never_go_through_ocr() -> None:
    ocr = FakeOcr()
    result, fake, _ = await _run(ocr, data=_pdf())
    assert result.document is not None
    assert result.document.page_kinds == {1: "text"}
    assert ocr.calls == 0
    assert _images_sent(fake) == 0


# ---------------------------------------------------------------- real OCR on every scan profile
@pytest.mark.ocr
@pytest.mark.parametrize("profile", ["clean", "phone_photo", "fax", "stamped", "low_dpi"])
async def test_real_ocr_end_to_end_on_every_scan_profile(profile: str) -> None:
    pytest.importorskip("rapidocr")
    scan_sim = pytest.importorskip("tools.synth.scan_sim")
    from bhoot.adapters.ocr import OpenCvScanPreprocessor, RapidOcrEngine

    scanned = scan_sim.ScanSimulator(scan_sim.PROFILES[profile], seed=11).simulate(_pdf())
    result, _fake, _ = await _run(
        RapidOcrEngine(), preprocessor=OpenCvScanPreprocessor(), data=scanned
    )
    assert result.status is JobStatus.SUCCEEDED, (profile, result.review_reasons)
    assert result.data == {
        "account_number": "718347936524",
        "closing_balance": Decimal("207815.11"),
    }
    assert result.document is not None
    assert result.document.page_kinds == {1: "ocr_text"}
    for name in ("account_number", "closing_balance"):
        assert result.fields[name].bbox is not None
    assert result.versions.ocr_engine is not None
    assert result.versions.ocr_engine.startswith("rapidocr-")
