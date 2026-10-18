"""Stackable LLMClient decorators. Each one adds one concern; all of them are LLMClients.

Order used by bootstrap (outermost first):
Tracing → Caching → ConcurrencyLimited → RateLimited → Retrying → LiteLLMClient

* a cache hit skips the concurrency slot, the rate limit, retries and the provider
* only real provider calls consume rate-limit budget
* retries sit closest to the network
"""

from __future__ import annotations

import time
from decimal import Decimal

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from bhoot.adapters.llm.rate_limiter import ConcurrencyLimiter, RateLimiter
from bhoot.domain.errors import ProviderTransientError, RateLimitedError
from bhoot.ports.llm import LLMClient, LLMRequest, LLMResponse, ModelCapabilities
from bhoot.ports.storage import LLMCache, Tracer


class _Wrapper:
    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.inner.capabilities(model)

    def count_tokens(self, model: str, text: str) -> int | None:
        return self.inner.count_tokens(model, text)


class RetryingLLM(_Wrapper):
    """Retries transient provider errors and rate limits with exponential backoff and jitter.
    Authentication and bad-request errors are never retried."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        max_attempts: int = 5,
        initial_s: float = 1.0,
        max_s: float = 30.0,
    ) -> None:
        super().__init__(inner)
        self.max_attempts = max_attempts
        self.initial_s = initial_s
        self.max_s = max_s

    async def complete(self, request: LLMRequest) -> LLMResponse:
        retrying = AsyncRetrying(
            retry=retry_if_exception_type((ProviderTransientError, RateLimitedError)),
            stop=stop_after_attempt(self.max_attempts),
            # exponential backoff plus jitter proportional to the base delay
            wait=wait_exponential(multiplier=self.initial_s, max=self.max_s)
            + wait_random(0, self.initial_s),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self.inner.complete(request)
        raise AssertionError("unreachable")  # pragma: no cover


class RateLimitedLLM(_Wrapper):
    def __init__(self, inner: LLMClient, limiter: RateLimiter) -> None:
        super().__init__(inner)
        self.limiter = limiter

    async def complete(self, request: LLMRequest) -> LLMResponse:
        estimate = request.max_tokens or 0
        for message in request.messages:
            for part in message.parts:
                estimate += len(part.text or "") // 4 + (800 if part.type == "image" else 0)
        await self.limiter.acquire(request.model, estimate)
        return await self.inner.complete(request)


class ConcurrencyLimitedLLM(_Wrapper):
    def __init__(self, inner: LLMClient, limiter: ConcurrencyLimiter) -> None:
        super().__init__(inner)
        self.limiter = limiter

    async def complete(self, request: LLMRequest) -> LLMResponse:
        async with self.limiter.slot():
            return await self.inner.complete(request)


class CachingLLM(_Wrapper):
    """Content-addressed response cache. Only successfully parsed responses are stored, and a
    hit costs nothing."""

    def __init__(self, inner: LLMClient, cache: LLMCache) -> None:
        super().__init__(inner)
        self.cache = cache

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key()
        started = time.perf_counter()
        hit = await self.cache.get(key)
        if hit is not None:
            return hit.model_copy(
                update={
                    "cache_hit": True,
                    "cost_usd": Decimal("0"),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        response = await self.inner.complete(request)
        if response.parsed is not None or request.response_format is None:
            await self.cache.put(key, response)
        return response


class TracingLLM(_Wrapper):
    def __init__(self, inner: LLMClient, tracer: Tracer) -> None:
        super().__init__(inner)
        self.tracer = tracer

    async def complete(self, request: LLMRequest) -> LLMResponse:
        with self.tracer.span(
            "llm.complete", model=request.model, **dict(request.metadata)
        ) as span:
            response = await self.inner.complete(request)
            span.set("prompt_tokens", response.usage.prompt_tokens)
            span.set("completion_tokens", response.usage.completion_tokens)
            span.set("cache_hit", response.cache_hit)
            span.set("latency_ms", response.latency_ms)
            return response
