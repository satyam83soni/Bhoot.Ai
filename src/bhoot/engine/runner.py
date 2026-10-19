"""Run one job through the extraction graph and turn the outcome into a result.

Problems with the document, the time limit or an engine bug come back as a FAILED result (or
are re-raised with ``raise_on_failure``); developer mistakes (configuration, authentication,
registry) are raised, because no result can fix them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from bhoot.domain.cost import CostReport
from bhoot.domain.errors import (
    DocumentError,
    EngineError,
    JobNotFoundError,
    JobTimeoutError,
    LLMError,
    ProviderAuthError,
)
from bhoot.domain.job import JobSpec
from bhoot.domain.result import ExtractionResult
from bhoot.engine.graph.builder import build_graph, recursion_limit
from bhoot.engine.graph.serde import engine_serializer
from bhoot.engine.graph.spec import GRAPH_SPECS
from bhoot.engine.results import ResultAssembler
from bhoot.engine.services import EngineServices, JobContext
from bhoot_graph import BaseCheckpointSaver, CompiledGraph, MemorySaver

RECOVERABLE = (DocumentError, EngineError, LLMError, JobTimeoutError)


def default_checkpointer() -> BaseCheckpointSaver:
    return MemorySaver(engine_serializer())


class EngineRunner:
    def __init__(
        self,
        services: EngineServices,
        checkpointer: BaseCheckpointSaver | None = None,
        results: ResultAssembler | None = None,
    ) -> None:
        self.services = services
        self.checkpointer = checkpointer or default_checkpointer()
        self.results = results or ResultAssembler()

    def graph(self, job: JobSpec, attached_rules: tuple[str, ...] = ()) -> CompiledGraph:
        spec = GRAPH_SPECS.get(self.services.settings.engine.graph)
        return build_graph(
            self.services,
            JobContext(job, attached_rules),
            spec=spec,
            checkpointer=self.checkpointer,
        )

    def config(self, job: JobSpec) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": job.job_id},
            "recursion_limit": recursion_limit(job.mode.max_rounds),
            "max_concurrency": self.services.settings.concurrency.max_parallel_llm_calls,
        }

    async def run(
        self, job: JobSpec, *, attached_rules: tuple[str, ...] = ()
    ) -> ExtractionResult[Any]:
        initial = {"job_id": job.job_id, "round": 0, "field_states": {}, "cost": CostReport()}
        return await self._drive(job, initial, attached_rules)

    async def resume(
        self, job: JobSpec, *, attached_rules: tuple[str, ...] = ()
    ) -> ExtractionResult[Any]:
        """Continue a job from its last saved step (after a crash or a timeout)."""
        graph = self.graph(job, attached_rules)
        snapshot = await graph.aget_state(self.config(job))
        if snapshot is None:
            raise JobNotFoundError(f"no saved state for job {job.job_id}")
        return await self._drive(
            job, None, attached_rules, already_spent=snapshot.values.get("cost")
        )

    async def stream(
        self, job: JobSpec, *, attached_rules: tuple[str, ...] = ()
    ) -> AsyncIterator[tuple[str, Any]]:
        """Progress as it happens: ``("node", {...})`` per finished step, then
        ``("result", ExtractionResult)``."""
        graph = self.graph(job, attached_rules)
        self.services.budget.open(job.job_id)
        state: dict[str, Any] = {}
        initial = {"job_id": job.job_id, "round": 0, "field_states": {}, "cost": CostReport()}
        try:
            async with asyncio.timeout(job.options.timeout_s):
                async for mode, chunk in graph.astream(
                    initial, self.config(job), stream_mode=["updates", "values"]
                ):
                    if mode == "values":
                        state = chunk
                    else:
                        for node, update in chunk.items():
                            yield (
                                "node",
                                {
                                    "node": node,
                                    "round": state.get("round", 0),
                                    "keys": sorted(update),
                                },
                            )
            yield "result", self._finish(job, state)
        except TimeoutError as exc:
            yield (
                "result",
                self._fail(
                    job, JobTimeoutError(f"job exceeded {job.options.timeout_s}s"), state, exc
                ),
            )
        except RECOVERABLE as exc:
            yield "result", self._fail(job, exc, state, exc)
        finally:
            self.services.budget.close(job.job_id)

    # ---------------------------------------------------------------- internals
    async def _drive(
        self,
        job: JobSpec,
        initial: Mapping[str, Any] | None,
        attached_rules: tuple[str, ...],
        already_spent: CostReport | None = None,
    ) -> ExtractionResult[Any]:
        graph = self.graph(job, attached_rules)
        spent = already_spent.total_usd if already_spent is not None else None
        self.services.budget.open(
            job.job_id, spent if spent is not None else CostReport().total_usd
        )
        state: Mapping[str, Any] = {}
        try:
            async with asyncio.timeout(job.options.timeout_s):
                state = await graph.ainvoke(initial, self.config(job))
            return self._finish(job, state)
        except ProviderAuthError:
            raise
        except TimeoutError as exc:
            error = JobTimeoutError(
                f"job exceeded {job.options.timeout_s}s (resume it to continue)"
            )
            return self._fail(job, error, await self._saved(graph, job), exc)
        except RECOVERABLE as exc:
            return self._fail(job, exc, await self._saved(graph, job), exc)
        finally:
            self.services.budget.close(job.job_id)

    async def _saved(self, graph: CompiledGraph, job: JobSpec) -> Mapping[str, Any]:
        snapshot = await graph.aget_state(self.config(job))
        return snapshot.values if snapshot is not None else {}

    def _finish(self, job: JobSpec, state: Mapping[str, Any]) -> ExtractionResult[Any]:
        return self.results.build(job, state, include_trace=job.options.include_trace)

    def _fail(
        self, job: JobSpec, error: BaseException, state: Mapping[str, Any], cause: BaseException
    ) -> ExtractionResult[Any]:
        if job.options.raise_on_failure:
            raise cause
        return self.results.failed(job, error, state)
