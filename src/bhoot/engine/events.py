"""Engine events: a tiny synchronous publish/subscribe bus.

Nodes publish what happened (a node started or finished, a model was called, a field was
decided, the budget ran out); tracing, metrics and progress streams subscribe. A subscriber
that raises is logged and skipped: observers never break a job.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeVar

from bhoot.domain.cost import Usage
from bhoot.domain.decisions import DecisionKind
from bhoot.domain.modes import Tier

log = logging.getLogger("bhoot.events")


@dataclass(frozen=True, slots=True)
class NodeStarted:
    job_id: str
    node: str
    round: int


@dataclass(frozen=True, slots=True)
class NodeFinished:
    job_id: str
    node: str
    round: int
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LLMCalled:
    job_id: str
    work_item_id: str
    model: str
    tier: Tier
    usage: Usage
    cost_usd: Decimal
    latency_ms: int
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class FieldDecided:
    job_id: str
    path: str
    decision: DecisionKind
    rule: str
    reason: str
    round: int


@dataclass(frozen=True, slots=True)
class BudgetExhausted:
    job_id: str
    work_item_id: str
    estimate_usd: Decimal
    remaining_usd: Decimal | None


Event = NodeStarted | NodeFinished | LLMCalled | FieldDecided | BudgetExhausted
E = TypeVar("E")


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[tuple[type[Any] | None, Callable[[Any], None]]] = []

    def subscribe(self, handler: Callable[[E], None], event_type: type[E] | None = None) -> None:
        """Call ``handler`` for every event, or only for ``event_type`` events."""
        self._subscribers.append((event_type, handler))

    def publish(self, event: Event) -> None:
        for event_type, handler in self._subscribers:
            if event_type is not None and not isinstance(event, event_type):
                continue
            try:
                handler(event)
            except Exception:
                log.exception("event subscriber %r failed on %s", handler, type(event).__name__)
