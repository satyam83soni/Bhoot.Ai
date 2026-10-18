"""Null Objects and simple defaults: every port always has an implementation, so the engine
never checks ``if x is not None``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from bhoot.domain.job import JobSpec
from bhoot.domain.result import ExtractionResult
from bhoot.domain.training import TrainingRecord
from bhoot.ports.llm import LLMResponse


class NullLLMCache:
    async def get(self, key: str) -> LLMResponse | None:
        return None

    async def put(self, key: str, response: LLMResponse) -> None:
        return None


class InMemoryLLMCache:
    """Per-process response cache (tests, notebooks). Persistent caches use SQLite."""

    def __init__(self) -> None:
        self.entries: dict[str, LLMResponse] = {}

    async def get(self, key: str) -> LLMResponse | None:
        return self.entries.get(key)

    async def put(self, key: str, response: LLMResponse) -> None:
        self.entries[key] = response


class NullResultCache:
    async def get(self, key: str) -> ExtractionResult[Any] | None:
        return None

    async def put(self, key: str, result: ExtractionResult[Any], ttl_s: int | None = None) -> None:
        return None

    async def clear(self) -> int:
        return 0


class NullTrainingSink:
    async def write(self, records: Sequence[TrainingRecord]) -> None:
        return None


class NullDebugSink:
    async def dump(self, job: JobSpec, node: str, payload: Mapping[str, Any]) -> None:
        return None


class NoopSpan:
    def set(self, key: str, value: Any) -> None:
        return None

    def __enter__(self) -> NoopSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class NoopTracer:
    def span(self, name: str, **attrs: Any) -> NoopSpan:
        return NoopSpan()


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
