"""Per-model rate limiting (requests and tokens per minute) and a global concurrency limit.

Both sit behind tiny protocols so the server can later swap in shared, cross-replica limiters
without touching the LLM decorators.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol


class RateLimiter(Protocol):
    async def acquire(self, model: str, tokens: int) -> None: ...


class ConcurrencyLimiter(Protocol):
    def slot(self) -> AbstractAsyncContextManager[None]: ...


class _Bucket:
    def __init__(self, per_minute: float, clock: Callable[[], float]) -> None:
        self.capacity = per_minute
        self.level = per_minute
        self.rate = per_minute / 60.0
        self.clock = clock
        self.stamp = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.level = min(self.capacity, self.level + (now - self.stamp) * self.rate)
        self.stamp = now

    def wait_needed(self, amount: float) -> float:
        """Seconds until ``amount`` is available (0 if it is available now). Takes nothing."""
        self._refill()
        amount = min(amount, self.capacity)  # one huge request can never be "too big" to send
        return 0.0 if self.level >= amount else (amount - self.level) / self.rate

    def take(self, amount: float) -> None:
        self.level -= min(amount, self.capacity)


class AsyncTokenBucket:
    """Token buckets per model for requests/min and tokens/min. Unknown models are unlimited."""

    def __init__(
        self,
        limits: Mapping[str, tuple[int | None, int | None]],  # model -> (rpm, tpm)
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self._limits = dict(limits)
        self._clock = clock
        self._sleep = sleep
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = asyncio.Lock()

    def _bucket(self, model: str, kind: str, per_minute: int) -> _Bucket:
        key = (model, kind)
        if key not in self._buckets:
            self._buckets[key] = _Bucket(float(per_minute), self._clock)
        return self._buckets[key]

    async def acquire(self, model: str, tokens: int) -> None:
        rpm, tpm = self._limits.get(model, (None, None))
        while True:
            async with self._lock:
                wanted = []
                if rpm:
                    wanted.append((self._bucket(model, "rpm", rpm), 1.0))
                if tpm:
                    wanted.append((self._bucket(model, "tpm", tpm), float(max(tokens, 1))))
                delay = max((bucket.wait_needed(amount) for bucket, amount in wanted), default=0.0)
                if delay <= 0:  # take from all buckets only when every limit allows it
                    for bucket, amount in wanted:
                        bucket.take(amount)
                    return
            await self._sleep(delay)


class SemaphoreLimiter:
    """Process-wide cap on in-flight LLM calls (bulkhead)."""

    def __init__(self, max_concurrent: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def slot(self) -> AbstractAsyncContextManager[None]:
        return _Slot(self._semaphore)


class _Slot:
    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self._semaphore.release()
