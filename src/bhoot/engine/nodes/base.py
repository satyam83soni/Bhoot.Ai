"""Common node behaviour (Template Method): timing, events and error wrapping.

A node's ``run`` returns a partial state update. ``BhootError``s pass through unchanged (the
runner maps them to results); anything else is a bug and is wrapped in ``EngineError`` naming
the node, so a failure always says where it happened.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from bhoot.domain.errors import BhootError, EngineError
from bhoot.engine.events import NodeFinished, NodeStarted
from bhoot.engine.services import EngineServices, JobContext

Update = dict[str, Any]


class BaseNode:
    name = "node"

    def __init__(self, services: EngineServices, ctx: JobContext) -> None:
        self.services = services
        self.ctx = ctx

    async def __call__(self, state: Mapping[str, Any]) -> Update:
        round_ = int(state.get("round", 0)) if isinstance(state, Mapping) else 0
        events = self.services.events
        events.publish(NodeStarted(self.ctx.job_id, self.name, round_))
        started = time.perf_counter()
        error: str | None = None
        try:
            update = await self.run(state)
        except BhootError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise EngineError(f"{self.name} failed: {error}", node=self.name) from exc
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            events.publish(NodeFinished(self.ctx.job_id, self.name, round_, elapsed, error))
        update.setdefault("timings_ms", {})[self.timing_key(state)] = round(elapsed, 3)
        return update

    def timing_key(self, state: Mapping[str, Any]) -> str:
        return self.name

    async def run(self, state: Mapping[str, Any]) -> Update:
        raise NotImplementedError
