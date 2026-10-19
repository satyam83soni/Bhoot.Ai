"""The whole engine on a real PDF with a scripted model: extraction, repairs, escalation,
cross-checks, budget limits, absent fields, crash-and-resume, timeouts and streaming."""

from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal
from io import BytesIO
from typing import Any

import pytest
from reportlab.pdfgen.canvas import Canvas

from bhoot.adapters.llm.fake import FakeLLMClient
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
from bhoot.config.settings import BhootSettings
from bhoot.domain.errors import JobTimeoutError
from bhoot.domain.job import ExtractRequest, JobSpec
from bhoot.domain.modes import Mode
from bhoot.domain.options import ExtractOptions
from bhoot.domain.result import JobStatus
from bhoot.domain.schema_dsl import Schema, field
from bhoot.domain.schema_introspect import introspect
from bhoot.engine.events import FieldDecided, LLMCalled
from bhoot.engine.job_factory import JobFactory
from bhoot.engine.runner import EngineRunner
from bhoot.engine.services import EngineServices


class Statement(Schema):
    account_number: str = field(criticality="critical", description="Account number as printed")
    closing_balance: Decimal = field(criticality="critical")
    holder: str | None = field(default=None, description="Account holder name")


PAGE = ["Statement of Account", "Account Number: 718347936524", "Closing Balance: 1,250.00"]


def _pdf(lines: list[str]) -> bytes:
    buf = BytesIO()
    canvas = Canvas(buf)
    for n, line in enumerate(lines):
        canvas.drawString(72, 720 - 20 * n, line)
    canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _answer(
    account: str = "718347936524", closing: str = "1,250.00", holder: str | None = None
) -> dict[str, Any]:
    return {
        "account_number": {"value": account, "quote": f"Account Number: {account}", "page": 1},
        "closing_balance": {"value": closing, "quote": "Closing Balance: 1,250.00", "page": 1},
        "holder": {"value": holder, "quote": None, "page": None},
    }


class Harness:
    def __init__(self, *, budget: float | None = 0.25, cross: bool = True, mode: Mode = Mode.BALANCED,
                 options: ExtractOptions | None = None) -> None:  # fmt: skip
        price = {"input_cost_per_mtok": 1.0, "output_cost_per_mtok": 2.0}
        models = ModelTierConfig(
            cheap=ModelSpec(model="openai/cheap", **price),
            strong=ModelSpec(model="openai/strong", **price),
            cross=ModelSpec(model="anthropic/cross", **price) if cross else None,
        )
        self.settings = BhootSettings(models=models, default_budget_usd=budget)
        self.llm = FakeLLMClient()
        self.blobs = MemoryBlobStore()
        self.services = EngineServices(
            settings=self.settings,
            llm=self.llm,
            blobs=self.blobs,
            parsers=MimeParserSelector([PdfiumDocumentParser(), ImageDocumentParser()]),
            renderer=DefaultPageRenderer(),
            preprocessor=PillowPreprocessor(),
            crops=PillowCropRenderer(),
        )
        self.events: list[object] = []
        self.services.events.subscribe(self.events.append)
        self.runner = EngineRunner(self.services)
        self.mode = mode
        self.options = options or ExtractOptions(include_trace=True)

    async def job(self, pdf: bytes | None = None) -> JobSpec:
        data = pdf or _pdf(PAGE)
        ref = await self.blobs.put(data, namespace="inputs", suffix=".pdf")
        req = ExtractRequest(
            doc_sha256=hashlib.sha256(data).hexdigest(), input_ref=ref, schema=introspect(Statement),
            mode=self.mode, options=self.options,
        )  # fmt: skip
        return JobFactory(self.settings).create(req)

    def calls(self, **meta: str) -> list[Any]:
        return [c for c in self.llm.calls if all(c.metadata.get(k) == v for k, v in meta.items())]


# ---------------------------------------------------------------- happy path and repairs
async def test_clean_extraction_is_verified_and_explained() -> None:
    h = Harness()
    h.llm.when(kind="extract").respond(_answer())
    result = await h.runner.run(await h.job())
    assert result.status is JobStatus.SUCCEEDED, result.review_reasons
    assert result.data["closing_balance"] == Decimal("1250.00")
    assert result.data["holder"] is None
    closing = result.fields["closing_balance"]
    assert closing.status == "verified"
    assert closing.page == 1
    assert closing.bbox is not None
    assert closing.quote == "Closing Balance: 1,250.00"
    assert result.fields["holder"].status == "absent"
    assert result.cost.llm_calls == 1
    assert result.cost.total_usd > 0
    assert result.versions.schema_ref == "statement@0"
    assert "evidence   page 1" in result.explain("closing_balance").text
    assert any(isinstance(e, LLMCalled) for e in h.events)
    request = h.llm.calls[0]
    assert request.model == "openai/cheap"
    assert request.response_format is not None
    assert request.response_format.strict


async def test_a_wrong_value_is_repaired_with_feedback() -> None:
    h = Harness()
    h.llm.when(kind="extract").respond(_answer(closing="1,520.00"))  # quote right, value wrong
    h.llm.when(kind="repair").respond(
        lambda req: {
            "closing_balance": {
                "value": "1,250.00",
                "quote": "Closing Balance: 1,250.00",
                "page": 1,
            }
        }
    )
    result = await h.runner.run(await h.job())
    assert result.status is JobStatus.SUCCEEDED
    assert result.data["closing_balance"] == Decimal("1250.00")
    repair = h.calls(kind="repair")[0]
    assert repair.metadata["paths"] == "closing_balance"
    feedback = repair.messages[-1].parts[0].text or ""
    assert "You returned closing_balance = '1,520.00'" in feedback
    assert result.fields["closing_balance"].attempts == 2
    assert result.fields["closing_balance"].trace is not None
    assert "r0:rule8:repair" in result.fields["closing_balance"].trace.decisions


async def test_repeated_failure_escalates_to_the_strong_model_then_review() -> None:
    h = Harness()
    h.llm.when(kind="extract").respond(_answer(closing="9,999.00"))
    h.llm.when(kind="repair").respond(
        lambda req: {
            "closing_balance": {
                "value": "9,999.00",
                "quote": "Closing Balance: 9,999.00",
                "page": 1,
            }
        }
    )
    result = await h.runner.run(await h.job())
    assert result.status is JobStatus.NEEDS_REVIEW
    assert result.fields["closing_balance"].status == "needs_review"
    assert result.fields["account_number"].status == "verified"
    assert [c.model for c in h.llm.calls].count("openai/strong") == 1
    assert any("closing_balance" in r for r in result.review_reasons)
    decided = [e for e in h.events if isinstance(e, FieldDecided) and e.path == "closing_balance"]
    assert [e.decision.value for e in decided] == ["repair", "repair", "escalate", "review"]


# ---------------------------------------------------------------- cross-checks
@pytest.mark.parametrize(
    ("second", "status"), [("1,250.00", "verified"), ("1,205.00", "needs_review")]
)
async def test_critical_fields_are_cross_checked_in_max_accuracy(second: str, status: str) -> None:
    h = Harness(mode=Mode.MAX_ACCURACY)
    h.llm.when(kind="extract").respond(_answer())
    h.llm.when(kind="crosscheck").respond(
        lambda req: {
            "account_number": {
                "value": "718347936524",
                "quote": "Account Number: 718347936524",
                "page": 1,
            },
            "closing_balance": {"value": second, "quote": f"Closing Balance: {second}", "page": 1},
        }
    )
    result = await h.runner.run(await h.job())
    assert result.fields["closing_balance"].status == status
    cross = h.calls(kind="crosscheck")
    assert cross
    assert cross[0].model == "anthropic/cross"
    assert all("You returned" not in (m.parts[0].text or "") for m in cross[0].messages)


async def test_without_a_cross_model_critical_fields_are_accepted_unconfirmed() -> None:
    h = Harness(mode=Mode.MAX_ACCURACY, cross=False)
    h.llm.when(kind="extract").respond(_answer())
    h.llm.when(kind="repair").respond(_answer())
    result = await h.runner.run(await h.job())
    assert result.status is JobStatus.SUCCEEDED
    assert not h.calls(kind="crosscheck")


# ---------------------------------------------------------------- budget and absent fields
async def test_no_budget_sends_everything_to_review_without_calls() -> None:
    h = Harness(budget=0.0000001)
    h.llm.when().respond(_answer())
    result = await h.runner.run(await h.job())
    assert h.llm.calls == []
    assert result.status is JobStatus.NEEDS_REVIEW
    assert result.cost.total_usd == 0
    assert result.fields["account_number"].status == "needs_review"


# ---------------------------------------------------------------- failures
async def test_a_crash_mid_job_fails_cleanly_and_resumes_without_repeating_calls() -> None:
    h = Harness()
    h.llm.when(kind="extract").fail(RuntimeError("worker died"), times=1)
    h.llm.when(kind="extract").respond(_answer())
    job = await h.job()
    failed = await h.runner.run(job)
    assert failed.status is JobStatus.FAILED
    assert "worker died" in (failed.error or "")
    resumed = await h.runner.resume(job)
    assert resumed.status is JobStatus.SUCCEEDED
    assert len(h.llm.calls) == 2  # the failed call, then the one retry; nothing else repeated


async def test_timeouts_fail_the_job_or_raise_when_asked() -> None:
    h = Harness(options=ExtractOptions(timeout_s=0.05))

    original = h.llm.complete

    async def delayed(request: Any) -> Any:
        await asyncio.sleep(1)
        return await original(request)

    h.llm.complete = delayed  # type: ignore[method-assign]
    h.llm.when().respond(_answer())
    result = await h.runner.run(await h.job())
    assert result.status is JobStatus.FAILED
    assert "exceeded" in (result.error or "")

    strict = Harness(options=ExtractOptions(timeout_s=0.05, raise_on_failure=True))
    strict.llm.complete = delayed  # type: ignore[method-assign]
    with pytest.raises((TimeoutError, JobTimeoutError)):
        await strict.runner.run(await strict.job())


async def test_streaming_reports_progress_then_the_result() -> None:
    h = Harness()
    h.llm.when(kind="extract").respond(_answer())
    events = [e async for e in h.runner.stream(await h.job())]
    nodes = [payload["node"] for kind, payload in events if kind == "node"]
    assert nodes[:3] == ["load", "classify", "plan"]
    assert "llm_extract" in nodes
    assert events[-1][0] == "result"
    assert events[-1][1].status is JobStatus.SUCCEEDED
