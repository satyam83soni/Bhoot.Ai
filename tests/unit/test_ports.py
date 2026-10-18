"""Ports: request hashing, OCR page helpers, and that simple fakes satisfy the protocols."""

from __future__ import annotations

from datetime import UTC, datetime

from bhoot.domain.document import BBox
from bhoot.ports import (
    Clock,
    ContentPart,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ModelCapabilities,
    OcrLine,
    OcrPage,
    OcrWord,
    ResponseFormat,
)


def _request(**kw: object) -> LLMRequest:
    args: dict[str, object] = {
        "model": "openai/x",
        "messages": (
            LLMMessage.text("system", "rules"),
            LLMMessage(role="user", parts=(ContentPart(type="image", image_ref="blob/abc.png"),)),
        ),
        "response_format": ResponseFormat(name="out", json_schema={"type": "object"}),
    }
    args.update(kw)
    return LLMRequest(**args)  # type: ignore[arg-type]


def test_cache_key_ignores_metadata_but_not_content() -> None:
    a = _request(metadata={"job": "1"})
    b = _request(metadata={"job": "2"})
    assert a.cache_key() == b.cache_key()
    assert a.cache_key() != _request(temperature=0.5).cache_key()
    assert a.cache_key() != _request(model="openai/y").cache_key()
    assert a.cache_key().startswith("sha256:")


def test_ocr_page_helpers() -> None:
    box = BBox(0, 0, 10, 10)
    page = OcrPage(
        lines=(OcrLine((OcrWord("a", box, 0.9), OcrWord("b", box, 0.7)), box),),
        image_width=100,
        image_height=100,
        engine="fake",
    )
    assert page.word_count == 2
    assert abs(page.mean_confidence - 0.8) < 1e-9
    assert OcrPage((), 1, 1, "fake").mean_confidence == 0.0


class _FakeLLM:
    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="{}", model=request.model)

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(vision=True, json_schema=True, strict_schema=True)

    def count_tokens(self, model: str, text: str) -> int | None:
        return None


class _FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


def test_simple_fakes_satisfy_the_protocols() -> None:
    assert isinstance(_FakeLLM(), LLMClient)
    assert isinstance(_FixedClock(), Clock)
    assert not isinstance(object(), LLMClient)
