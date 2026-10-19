"""The public ``Bhoot`` facade: inputs, typed results, sync and async use, streaming, many
documents, resume, caching and the errors developers see."""

from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from reportlab.pdfgen.canvas import Canvas

import bhoot
from bhoot import Bhoot, ExtractOptions, JobStatus, Mode, Schema, field
from bhoot.adapters.llm.fake import FakeLLMClient
from bhoot.api import facade
from bhoot.config.models import ModelSpec, ModelTierConfig
from bhoot.config.settings import BhootSettings, LimitsSettings
from bhoot.domain.errors import DocumentError, FileTooLargeError, JobNotFoundError, ModelConfigError


class Invoice(Schema):
    number: str = field(criticality="critical")
    total: Decimal = field(criticality="critical")


def _pdf(number: str = "INV-0042", total: str = "1,250.00") -> bytes:
    buf = BytesIO()
    canvas = Canvas(buf)
    canvas.drawString(72, 720, f"Invoice No: {number}")
    canvas.drawString(72, 700, f"Total Due: {total}")
    canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _answer(number: str = "INV-0042", total: str = "1,250.00") -> dict[str, Any]:
    return {
        "number": {"value": number, "quote": f"Invoice No: {number}", "page": 1},
        "total": {"value": total, "quote": f"Total Due: {total}", "page": 1},
    }


def _settings(**kw: Any) -> BhootSettings:
    models = ModelTierConfig(
        cheap=ModelSpec(model="openai/cheap", input_cost_per_mtok=1, output_cost_per_mtok=2),
        strong=ModelSpec(model="openai/strong"),
    )
    return BhootSettings(models=models, **kw)


def _bhoot(llm: FakeLLMClient | None = None, **kw: Any) -> tuple[Bhoot, FakeLLMClient]:
    fake = llm or FakeLLMClient()
    if not fake.rules:
        fake.when().respond(lambda req: _answer())
    return Bhoot(settings=_settings(**kw), llm=fake), fake


# ---------------------------------------------------------------- inputs and typed results
def test_extract_from_bytes_gives_a_typed_verified_result() -> None:
    bh, fake = _bhoot()
    with bh:
        result = bh.extract(_pdf(), schema=Invoice)
    assert result.status is JobStatus.SUCCEEDED
    invoice = result.typed
    assert isinstance(invoice, Invoice)
    assert invoice.total == Decimal("1250.00")
    assert result.fields["number"].quote == "Invoice No: INV-0042"
    assert len(fake.calls) == 1


def test_paths_and_open_files_work_too(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    path.write_bytes(_pdf())
    bh, _ = _bhoot()
    with bh, path.open("rb") as handle:
        assert bh.extract(path, schema=Invoice).succeeded
        assert bh.extract(str(path), schema=Invoice).succeeded
        assert bh.extract(handle, schema=Invoice).succeeded


def test_input_mistakes_are_explained(tmp_path: Path) -> None:
    bh, _ = _bhoot(limits=LimitsSettings(max_file_mb=1))
    with bh:
        with pytest.raises(DocumentError, match="no file at"):
            bh.extract(tmp_path / "missing.pdf", schema=Invoice)
        with pytest.raises(DocumentError, match="empty"):
            bh.extract(b"", schema=Invoice)
        with pytest.raises(FileTooLargeError, match="limit is 1 MB"):
            bh.extract(b"%PDF-" + b"0" * 2_000_000, schema=Invoice)
        with pytest.raises(TypeError, match="cannot read a document"):
            bh.extract(42, schema=Invoice)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="schema must be"):
            bh.extract(_pdf(), schema=int)  # type: ignore[arg-type]


def test_an_unreadable_file_is_a_failed_result_not_an_exception() -> None:
    bh, _ = _bhoot()
    with bh:
        result = bh.extract(b"this is not a pdf", schema=Invoice)
    assert result.status is JobStatus.FAILED
    assert "not supported" in (result.error or "")


def test_missing_model_config_says_how_to_fix_it() -> None:
    with Bhoot(settings=BhootSettings()) as bh, pytest.raises(ModelConfigError, match="models:"):
        bh.extract(_pdf(), schema=Invoice)


# ---------------------------------------------------------------- options
def test_options_modes_and_overrides_reach_the_job() -> None:
    bh, _ = _bhoot()
    with bh:
        result = bh.extract(
            _pdf(), schema=Invoice, mode="economy", include_trace=True, max_cost_usd=0.5
        )
        assert result.fields["total"].trace is not None
        plain = bh.extract(_pdf(), schema=Invoice, options=ExtractOptions(include_trace=False))
        assert plain.fields["total"].trace is None
    assert Mode("economy") is Mode.ECONOMY


def test_repeated_documents_hit_the_model_cache() -> None:
    bh, fake = _bhoot()
    with bh:
        first = bh.extract(_pdf(), schema=Invoice)
        second = bh.extract(_pdf(), schema=Invoice)
    assert len(fake.calls) == 1
    assert first.cost.llm_calls == 1
    assert second.cost.cache_hits == 1
    assert second.cost.total_usd == 0


# ---------------------------------------------------------------- async, many, stream, resume
async def test_async_api_streaming_and_many_documents() -> None:
    bh, _ = _bhoot()
    async with bh:
        result = await bh.aextract(_pdf(), schema=Invoice)
        assert result.typed.number == "INV-0042"
        events = [e async for e in bh.astream(_pdf(), schema=Invoice)]
        assert events[0] == (
            "node",
            {"node": "load", "round": 0, "keys": ["document", "timings_ms"]},
        )
        assert events[-1][0] == "result"
        assert events[-1][1].typed.total == Decimal("1250.00")
        many = await bh.aextract_many([_pdf(), b"not a pdf", _pdf()], schema=Invoice, concurrency=2)
    assert [r.status for r in many] == [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.SUCCEEDED]


async def test_sync_api_works_inside_a_running_event_loop() -> None:
    bh, _ = _bhoot()
    with bh:
        assert bh.extract(_pdf(), schema=Invoice).succeeded  # e.g. from a notebook cell


def test_a_failed_job_can_be_resumed() -> None:
    fake = FakeLLMClient()
    fake.when().fail(RuntimeError("worker died"), times=1)
    fake.when().respond(lambda req: _answer())
    bh, _ = _bhoot(fake)
    with bh:
        failed = bh.extract(_pdf(), schema=Invoice)
        assert failed.status is JobStatus.FAILED
        resumed = bh.resume(failed.job_id)
        assert resumed.typed.total == Decimal("1250.00")
        with pytest.raises(JobNotFoundError):
            bh.resume("job_unknown")


def test_module_level_shortcut_uses_the_shared_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    bh, _ = _bhoot()
    monkeypatch.setattr(facade, "_default", bh)
    try:
        assert bhoot.extract(_pdf(), schema=Invoice).succeeded
    finally:
        bh.close()
