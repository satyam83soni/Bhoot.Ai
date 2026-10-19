"""``Bhoot``: the one class developers use.

    from bhoot import Bhoot, Schema, field

    class Invoice(Schema):
        number: str = field(criticality="critical")
        total: Decimal = field(criticality="critical")

    bh = Bhoot()                      # reads bhoot.yaml from the project
    result = bh.extract("invoice.pdf", schema=Invoice)
    result.typed.total                # Decimal, verified on the page
    result.explain("total")           # where it came from and how it was checked

Every method has an async twin (``aextract``, ``aextract_many``, ``astream``, ``aresume``).
Use one style per ``Bhoot`` instance: the sync methods run on a private event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import AsyncIterator, Awaitable, Coroutine, Iterable
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypeVar

from pydantic import BaseModel

from bhoot.adapters.parsing.factory import sniff_mime
from bhoot.api.bootstrap import build_services
from bhoot.api.inputs import InputLike, read_input
from bhoot.config.loader import load_settings
from bhoot.config.settings import BhootSettings
from bhoot.domain.errors import JobNotFoundError
from bhoot.domain.job import ExtractRequest, JobSpec
from bhoot.domain.modes import Mode
from bhoot.domain.options import ExtractOptions
from bhoot.domain.pack import DocPack
from bhoot.domain.result import ExtractionResult
from bhoot.domain.schema import SchemaDefinition
from bhoot.domain.schema_introspect import introspect
from bhoot.engine.job_factory import JobFactory, ephemeral_pack
from bhoot.engine.runner import EngineRunner
from bhoot.engine.services import EngineServices
from bhoot.ports.llm import LLMClient
from bhoot_graph import BaseCheckpointSaver

T = TypeVar("T")
SchemaLike = type[BaseModel] | SchemaDefinition | DocPack
_SUFFIX = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
}


class Bhoot:
    def __init__(
        self,
        config: str | Path | None = None,
        *,
        project_dir: str | Path | None = None,
        settings: BhootSettings | None = None,
        llm: LLMClient | None = None,
        services: EngineServices | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        """``config`` is a path to ``bhoot.yaml`` (found automatically when omitted). ``llm``
        replaces the provider SDKs with your own client (still rate-limited, cached, retried)."""
        self.settings = settings or load_settings(config, project_dir=project_dir)
        self.services = services or build_services(self.settings, llm=llm)
        self._runner = EngineRunner(self.services, checkpointer)
        self._jobs: dict[str, tuple[JobSpec, type[BaseModel] | None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- async API
    async def aextract(
        self,
        source: InputLike,
        *,
        schema: SchemaLike,
        mode: Mode | str | None = None,
        max_cost_usd: float | Decimal | None = None,
        options: ExtractOptions | None = None,
        **option_overrides: Any,
    ) -> ExtractionResult[Any]:
        """Extract ``schema`` from one document. Never raises for a bad document or a
        provider outage (the result has status FAILED) unless ``raise_on_failure=True``."""
        job, model = await self._job(source, schema, mode, max_cost_usd, options, option_overrides)
        result = await self._runner.run(job)
        return result.bind(model) if model else result

    async def aextract_many(
        self,
        sources: Iterable[InputLike],
        *,
        schema: SchemaLike,
        concurrency: int | None = None,
        **kwargs: Any,
    ) -> list[ExtractionResult[Any]]:
        """Several documents at once, results in input order."""
        limit = asyncio.Semaphore(concurrency or self.settings.concurrency.max_parallel_docs)

        async def one(source: InputLike) -> ExtractionResult[Any]:
            async with limit:
                return await self.aextract(source, schema=schema, **kwargs)

        return list(await asyncio.gather(*(one(s) for s in sources)))

    async def astream(
        self,
        source: InputLike,
        *,
        schema: SchemaLike,
        mode: Mode | str | None = None,
        max_cost_usd: float | Decimal | None = None,
        options: ExtractOptions | None = None,
        **option_overrides: Any,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Progress while extracting: ``("node", {"node", "round", "keys"})`` events, then
        ``("result", ExtractionResult)``."""
        job, model = await self._job(source, schema, mode, max_cost_usd, options, option_overrides)
        async for kind, payload in self._runner.stream(job):
            if kind == "result" and model is not None:
                payload = payload.bind(model)
            yield kind, payload

    async def aresume(self, job_id: str) -> ExtractionResult[Any]:
        """Continue a job that failed or timed out, from its last saved step (same process)."""
        found = self._jobs.get(job_id)
        if found is None:
            raise JobNotFoundError(f"no job {job_id!r} was started by this Bhoot instance")
        job, model = found
        result = await self._runner.resume(job)
        return result.bind(model) if model else result

    async def aclose(self) -> None:
        self._jobs.clear()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- sync API
    def extract(
        self, source: InputLike, *, schema: SchemaLike, **kwargs: Any
    ) -> ExtractionResult[Any]:
        return self._sync(self.aextract(source, schema=schema, **kwargs))

    def extract_many(
        self, sources: Iterable[InputLike], *, schema: SchemaLike, **kwargs: Any
    ) -> list[ExtractionResult[Any]]:
        return self._sync(self.aextract_many(list(sources), schema=schema, **kwargs))

    def resume(self, job_id: str) -> ExtractionResult[Any]:
        return self._sync(self.aresume(job_id))

    def close(self) -> None:
        self._sync(self.aclose())
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._loop = self._thread = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def _sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run on this instance's private loop (works inside notebooks and running loops)."""
        if self._loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="bhoot-sync", daemon=True)
            thread.start()
            self._loop, self._thread = loop, thread
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ---------------------------------------------------------------- internals
    async def _job(
        self,
        source: InputLike,
        schema: SchemaLike,
        mode: Mode | str | None,
        max_cost_usd: float | Decimal | None,
        options: ExtractOptions | None,
        overrides: dict[str, Any],
    ) -> tuple[JobSpec, type[BaseModel] | None]:
        pack, model = _resolve_schema(schema)
        opts = options or ExtractOptions()
        if overrides:
            opts = ExtractOptions.model_validate({**opts.model_dump(), **overrides})
        data, name = await asyncio.to_thread(
            read_input, source, max_mb=self.settings.limits.max_file_mb
        )
        suffix = _SUFFIX.get(sniff_mime(data[:1024]), "")
        ref = await self.services.blobs.put(data, namespace="inputs", suffix=suffix)
        req = ExtractRequest(
            doc_sha256=hashlib.sha256(data).hexdigest(),
            input_ref=ref,
            schema=pack.schema,
            filename=name,
            mode=Mode(mode) if mode is not None else None,
            max_cost_usd=Decimal(str(max_cost_usd)) if max_cost_usd is not None else None,
            options=opts,
        )
        job = JobFactory(self.settings).create(req, pack)
        self._jobs[job.job_id] = (job, model)
        return job, model


def _resolve_schema(schema: SchemaLike) -> tuple[DocPack, type[BaseModel] | None]:
    if isinstance(schema, DocPack):
        return schema, None
    if isinstance(schema, SchemaDefinition):
        return ephemeral_pack(schema), None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return ephemeral_pack(introspect(schema)), schema
    raise TypeError(
        f"schema must be a Schema class, a SchemaDefinition or a DocPack, not {schema!r}"
    )


# ---------------------------------------------------------------- module-level shortcuts
_default: Bhoot | None = None
_default_lock = threading.Lock()


def _instance() -> Bhoot:
    global _default
    with _default_lock:
        if _default is None:
            _default = Bhoot()
        return _default


def extract(source: InputLike, *, schema: SchemaLike, **kwargs: Any) -> ExtractionResult[Any]:
    """``Bhoot().extract`` with a shared default instance (configured from bhoot.yaml)."""
    return _instance().extract(source, schema=schema, **kwargs)


def aextract(
    source: InputLike, *, schema: SchemaLike, **kwargs: Any
) -> Awaitable[ExtractionResult[Any]]:
    return _instance().aextract(source, schema=schema, **kwargs)
