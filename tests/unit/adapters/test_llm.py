"""LLM adapter: official-SDK provider clients (with fake SDK objects), router, decorators,
fake and replay clients."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bhoot.adapters.llm.anthropic_client import AnthropicClient
from bhoot.adapters.llm.decorators import (
    CachingLLM,
    ConcurrencyLimitedLLM,
    RateLimitedLLM,
    RetryingLLM,
    TracingLLM,
)
from bhoot.adapters.llm.fake import FakeLLMClient
from bhoot.adapters.llm.gemini_client import GeminiClient
from bhoot.adapters.llm.openai_client import OpenAIClient
from bhoot.adapters.llm.rate_limiter import AsyncTokenBucket, SemaphoreLimiter
from bhoot.adapters.llm.replay import CassetteMissError, ReplayLLMClient
from bhoot.adapters.llm.router import ModelRouter
from bhoot.adapters.null import InMemoryLLMCache, NoopTracer
from bhoot.config.models import ModelSpec, ModelTierConfig
from bhoot.domain.errors import (
    ModelCapabilityError,
    ModelConfigError,
    ProviderAuthError,
    ProviderTransientError,
    RateLimitedError,
)
from bhoot.ports.llm import ContentPart, LLMClient, LLMMessage, LLMRequest, ResponseFormat


def _request(model: str = "openai/cheap", **meta: str) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(LLMMessage.text("system", "rules"), LLMMessage.text("user", "page text")),
        response_format=ResponseFormat(name="out", json_schema={"type": "object"}),
        metadata=meta,
    )


class _Recorder:
    """Stands in for an SDK method: records kwargs, returns or raises the scripted outcome."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _error(name: str, **attrs: Any) -> Exception:
    exc = type(name, (Exception,), {})("boom")
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


class _Blobs:
    async def get(self, key: str) -> bytes:
        return b"\x89PNG fake"


def _image_request(model: str) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(
            LLMMessage.text("system", "rules"),
            LLMMessage(
                role="user",
                parts=(
                    ContentPart(type="text", text="read"),
                    ContentPart(type="image", image_ref="b/1.png"),
                ),
            ),
        ),
        response_format=ResponseFormat(name="out", json_schema={"type": "object"}),
    )


# ---------------------------------------------------------------- model config
def test_model_spec_provider_forms() -> None:
    short = ModelSpec(model="openai/gpt-x")
    assert (short.provider_name, short.model_name, short.model_id) == (
        "openai",
        "gpt-x",
        "openai/gpt-x",
    )
    explicit = ModelSpec(model="gpt-x", provider="openai")
    assert explicit.model_id == "openai/gpt-x"
    hf = ModelSpec(model="meta-llama/Llama-3-8B", provider="openai", api_base="http://gpu:8000/v1")
    assert hf.model_name == "meta-llama/Llama-3-8B"  # vLLM ids keep their slash
    with pytest.raises(ValueError, match="set provider"):
        ModelSpec(model="gpt-x")


# ---------------------------------------------------------------- OpenAI
def _openai_sdk(outcome: Any) -> tuple[Any, _Recorder]:
    create = _Recorder(outcome)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


def _openai_answer(content: str = '{"total": "1,250.00"}') -> Any:
    usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=40, prompt_tokens_details=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=usage,
    )


async def test_openai_call_and_answer() -> None:
    sdk, create = _openai_sdk(_openai_answer())
    spec = ModelSpec(model="openai/gpt-x", input_cost_per_mtok=1.0, output_cost_per_mtok=2.0)
    client = OpenAIClient(spec, sdk_client=sdk)
    response = await client.complete(_request("openai/gpt-x"))
    sent = create.calls[0]
    assert sent["model"] == "gpt-x"
    assert sent["temperature"] == 0.0
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["messages"][0] == {"role": "system", "content": "rules"}
    assert response.parsed == {"total": "1,250.00"}
    assert response.model == "openai/gpt-x"
    assert response.cost_usd == Decimal("0.00128")  # 1200 x $1/M + 40 x $2/M


async def test_openai_compatible_server_uses_json_mode() -> None:
    sdk, create = _openai_sdk(_openai_answer())
    spec = ModelSpec(model="my-model", provider="openai", api_base="http://gpu:8000/v1")
    client = OpenAIClient(spec, sdk_client=sdk)
    assert client.capabilities().dialect == "prompt_only"
    await client.complete(_request("openai/my-model"))
    assert create.calls[0]["response_format"] == {"type": "json_object"}
    assert (await client.complete(_request("openai/my-model"))).cost_usd == Decimal(
        "0"
    )  # no pricing


async def test_openai_images_and_errors() -> None:
    sdk, create = _openai_sdk(_openai_answer())
    await OpenAIClient(ModelSpec(model="openai/gpt-x"), _Blobs(), sdk_client=sdk).complete(
        _image_request("openai/gpt-x")
    )  # type: ignore[arg-type]
    parts = create.calls[0]["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    for name, expected in [("RateLimitError", RateLimitedError), ("APITimeoutError", ProviderTransientError),
                           ("AuthenticationError", ProviderAuthError), ("BadRequestError", ModelCapabilityError)]:  # fmt: skip
        sdk, _ = _openai_sdk(_error(name))
        with pytest.raises(expected):
            await OpenAIClient(ModelSpec(model="openai/gpt-x"), sdk_client=sdk).complete(
                _request("openai/gpt-x")
            )
    sdk, _ = _openai_sdk(_error("APIStatusError", status_code=503))
    with pytest.raises(ProviderTransientError):
        await OpenAIClient(ModelSpec(model="openai/gpt-x"), sdk_client=sdk).complete(
            _request("openai/gpt-x")
        )


# ---------------------------------------------------------------- Anthropic
def _anthropic_sdk(outcome: Any) -> tuple[Any, _Recorder]:
    create = _Recorder(outcome)
    return SimpleNamespace(messages=SimpleNamespace(create=create)), create


async def test_anthropic_uses_a_forced_tool_for_structured_output() -> None:
    answer = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="out", input={"total": "1,250.00"})],
        usage=SimpleNamespace(input_tokens=900, output_tokens=30, cache_read_input_tokens=0),
        stop_reason="tool_use",
    )
    sdk, create = _anthropic_sdk(answer)
    client = AnthropicClient(ModelSpec(model="anthropic/claude-x"), _Blobs(), sdk_client=sdk)  # type: ignore[arg-type]
    response = await client.complete(_image_request("anthropic/claude-x"))
    sent = create.calls[0]
    assert sent["system"] == "rules"
    assert sent["tool_choice"] == {"type": "tool", "name": "out"}
    assert sent["tools"][0]["input_schema"] == {"type": "object"}
    assert sent["max_tokens"] > 0
    assert sent["messages"][0]["content"][1]["source"]["media_type"] == "image/png"
    assert response.parsed == {"total": "1,250.00"}
    assert response.usage.prompt_tokens == 900


async def test_anthropic_errors() -> None:
    for name, expected in [("OverloadedError", ProviderTransientError), ("RateLimitError", RateLimitedError),
                           ("PermissionDeniedError", ProviderAuthError)]:  # fmt: skip
        sdk, _ = _anthropic_sdk(_error(name))
        with pytest.raises(expected):
            await AnthropicClient(ModelSpec(model="anthropic/c"), sdk_client=sdk).complete(
                _request("anthropic/c")
            )


# ---------------------------------------------------------------- Gemini
def _gemini_sdk(outcome: Any) -> tuple[Any, _Recorder]:
    generate = _Recorder(outcome)
    return SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    ), generate


async def test_gemini_uses_native_json_schema() -> None:
    answer = SimpleNamespace(
        text='{"total": "7"}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=500, candidates_token_count=10, cached_content_token_count=0
        ),
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    sdk, generate = _gemini_sdk(answer)
    client = GeminiClient(ModelSpec(model="gemini/flash"), _Blobs(), sdk_client=sdk)  # type: ignore[arg-type]
    response = await client.complete(_image_request("gemini/flash"))
    sent = generate.calls[0]
    assert sent["model"] == "flash"
    assert sent["config"]["system_instruction"] == "rules"
    assert sent["config"]["response_json_schema"] == {"type": "object"}
    assert sent["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/png"
    assert response.parsed == {"total": "7"}


async def test_gemini_errors_by_status_code() -> None:
    for code, expected in [(429, RateLimitedError), (503, ProviderTransientError), (403, ProviderAuthError),
                           (400, ModelCapabilityError)]:  # fmt: skip
        sdk, _ = _gemini_sdk(_error("ClientError", code=code))
        with pytest.raises(expected):
            await GeminiClient(ModelSpec(model="gemini/flash"), sdk_client=sdk).complete(
                _request("gemini/flash")
            )


# ---------------------------------------------------------------- router and keys
async def test_router_dispatches_by_provider_and_creates_clients_lazily() -> None:
    created: list[str] = []

    def factory(provider: str) -> Any:
        def make(spec: ModelSpec, blobs: Any, timeout: float) -> Any:
            created.append(spec.model_id)
            fake = FakeLLMClient()
            fake.when().respond({"provider": provider})
            fake.capabilities = lambda *_: fake.__class__.capabilities(fake, "x")  # type: ignore[method-assign]
            return fake

        return make

    tiers = ModelTierConfig(
        cheap=ModelSpec(model="openai/c"), strong=ModelSpec(model="anthropic/s")
    )
    router = ModelRouter(
        tiers,
        factories={
            "openai": factory("openai"),
            "anthropic": factory("anthropic"),
            "gemini": factory("gemini"),
        },
    )
    assert isinstance(router, LLMClient)
    assert (await router.complete(_request("anthropic/s"))).parsed == {"provider": "anthropic"}
    assert created == ["anthropic/s"]  # the OpenAI client was never needed, so never built
    with pytest.raises(ModelConfigError, match="not configured"):
        await router.complete(_request("gemini/unknown"))


def test_missing_key_and_missing_sdk_have_clear_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOT_SET_KEY", raising=False)
    with pytest.raises(ProviderAuthError, match="NOT_SET_KEY"):
        OpenAIClient(ModelSpec(model="openai/x", api_key_env="NOT_SET_KEY"))
    with pytest.raises(ModelConfigError, match=r"bhoot\[gemini\]"):
        load_sdk_or_explain("google.genai_does_not_exist")


def load_sdk_or_explain(module: str) -> Any:
    from bhoot.adapters.llm import common

    common.EXTRA_FOR[module] = "gemini"
    return common.load_sdk(module)


# ---------------------------------------------------------------- decorators
async def test_retry_recovers_from_transient_errors_but_not_auth() -> None:
    fake = FakeLLMClient()
    fake.when().fail(ProviderTransientError("blip"), times=2).when().respond({"ok": True})
    retrying = RetryingLLM(fake, max_attempts=5, initial_s=0.001, max_s=0.002)
    assert (await retrying.complete(_request())).parsed == {"ok": True}
    fake.assert_called(times=3)

    auth = FakeLLMClient()
    auth.when().fail(ProviderAuthError("bad key"))
    with pytest.raises(ProviderAuthError):
        await RetryingLLM(auth, initial_s=0.001).complete(_request())
    auth.assert_called(times=1)


async def test_retry_gives_up_after_max_attempts() -> None:
    fake = FakeLLMClient()
    fake.when().fail(RateLimitedError("429"))
    with pytest.raises(RateLimitedError):
        await RetryingLLM(fake, max_attempts=3, initial_s=0.001, max_s=0.002).complete(_request())
    fake.assert_called(times=3)


async def test_cache_hits_are_free_and_skip_the_provider() -> None:
    fake = FakeLLMClient()
    fake.when().respond({"total": "1"})
    cached = CachingLLM(fake, InMemoryLLMCache())
    first = await cached.complete(_request(job="1"))
    second = await cached.complete(_request(job="2"))  # metadata differs, content identical
    assert not first.cache_hit
    assert second.cache_hit
    assert second.cost_usd == Decimal("0")
    fake.assert_called(times=1)


async def test_concurrency_limit_is_respected() -> None:
    in_flight = peak = 0

    class _Slow(FakeLLMClient):
        async def complete(self, request: LLMRequest):  # type: ignore[no-untyped-def]
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return await super().complete(request)

    slow = _Slow()
    slow.when().respond({"ok": True})
    limited = ConcurrencyLimitedLLM(slow, SemaphoreLimiter(2))
    await asyncio.gather(*(limited.complete(_request()) for _ in range(6)))
    assert peak == 2


async def test_token_bucket_waits_when_the_budget_is_spent() -> None:
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    bucket = AsyncTokenBucket({"m": (2, None)}, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(3):
        await bucket.acquire("m", 10)
    assert len(slept) == 1
    assert slept[0] == pytest.approx(30.0)  # 2 per minute: the third waits half a minute
    await bucket.acquire("unlimited-model", 10**9)  # no limits configured: never waits
    assert len(slept) == 1


async def test_rate_limit_and_tracing_pass_through() -> None:
    fake = FakeLLMClient()
    fake.when().respond({"ok": 1})
    client = TracingLLM(RateLimitedLLM(fake, AsyncTokenBucket({})), NoopTracer())
    assert (await client.complete(_request())).parsed == {"ok": 1}
    assert client.capabilities("x").vision


# ---------------------------------------------------------------- fake and replay
async def test_fake_matches_on_metadata() -> None:
    fake = FakeLLMClient()
    fake.when(section="summary").respond({"closing_balance": "10"})
    fake.when(kind="repair", paths_contains="transactions[61]").respond({"rows": []})
    assert (await fake.complete(_request(section="summary"))).parsed == {"closing_balance": "10"}
    repair = _request(kind="repair", paths="transactions[60],transactions[61]")
    assert (await fake.complete(repair)).parsed == {"rows": []}
    with pytest.raises(AssertionError, match="no rule matches"):
        await fake.complete(_request(section="header"))


async def test_replay_records_then_replays(tmp_path: Path) -> None:
    cassette = tmp_path / "cassette.json"
    live = FakeLLMClient()
    live.when().respond({"recorded": True})
    recorder = ReplayLLMClient(cassette, inner=live, mode="record")
    await recorder.complete(_request())
    assert cassette.exists()

    replayer = ReplayLLMClient(cassette)
    answer = await replayer.complete(_request(job="other"))
    assert answer.parsed == {"recorded": True}
    assert answer.cache_hit
    with pytest.raises(CassetteMissError, match="re-record"):
        await replayer.complete(_request(model="openai/other"))
    with pytest.raises(ValueError, match="real client"):
        ReplayLLMClient(cassette, mode="record")
