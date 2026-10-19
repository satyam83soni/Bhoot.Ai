"""Registry, caches, blob storage, checkpoints, tracing, training log, debug dumps and clock.

Every port always has an implementation; "off" is a Null Object, so nodes never check for None.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from bhoot.domain.catalog import PackInfo, RegistryEntry
from bhoot.domain.example import Example
from bhoot.domain.job import JobSpec
from bhoot.domain.pack import DocPack
from bhoot.domain.prompt import PromptTemplate
from bhoot.domain.refs import Ref
from bhoot.domain.result import ExtractionResult
from bhoot.domain.schema import SchemaDefinition
from bhoot.domain.training import TrainingRecord
from bhoot.ports.llm import LLMResponse


@runtime_checkable
class Registry(Protocol):
    def resolve_schema(self, ref: Ref) -> SchemaDefinition: ...

    def resolve_prompt(self, ref: Ref) -> PromptTemplate: ...

    def resolve_examples(self, schema: SchemaDefinition) -> tuple[Example, ...]: ...

    def resolve_pack(self, schema_ref: Ref, prompt_ref: Ref | None) -> DocPack: ...

    def list_schemas(self) -> list[RegistryEntry]: ...

    def list_prompts(self) -> list[RegistryEntry]: ...

    def list_packs(self) -> list[PackInfo]: ...


@runtime_checkable
class ResultCache(Protocol):
    async def get(self, key: str) -> ExtractionResult[Any] | None: ...

    async def put(
        self, key: str, result: ExtractionResult[Any], ttl_s: int | None = None
    ) -> None: ...

    async def clear(self) -> int: ...


@runtime_checkable
class LLMCache(Protocol):
    async def get(self, key: str) -> LLMResponse | None: ...

    async def put(self, key: str, response: LLMResponse) -> None: ...


@runtime_checkable
class BlobStore(Protocol):
    async def put(self, data: bytes, *, namespace: str, suffix: str) -> str: ...  # returns the key

    async def get(self, key: str) -> bytes: ...

    async def exists(self, key: str) -> bool: ...


@runtime_checkable
class CheckpointerFactory(Protocol):
    async def create(self) -> object: ...  # a graph checkpoint saver (kept opaque here)


@runtime_checkable
class Span(Protocol):
    def set(self, key: str, value: Any) -> None: ...

    def __enter__(self) -> Span: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    def span(self, name: str, **attrs: Any) -> Span: ...


@runtime_checkable
class TrainingSink(Protocol):
    async def write(self, records: Sequence[TrainingRecord]) -> None: ...


@runtime_checkable
class DebugSink(Protocol):
    async def dump(self, job: JobSpec, node: str, payload: Mapping[str, Any]) -> None: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...
