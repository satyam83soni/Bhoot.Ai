"""Values nodes and routers return to steer the graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Send:
    """Run ``node`` in the next superstep with ``arg`` as its input instead of the state.

    Returned by a router (or in ``Command.goto``) once per item to fan out; each Send is its
    own task, so ten Sends to the same node run it ten times in parallel.
    """

    node: str
    arg: Any


@dataclass(frozen=True, slots=True)
class Command:
    """Returned by a node to update the state and choose where to go next in one step.

    ``goto`` adds to the node's static edges; it does not replace them. As the *input* of
    ``ainvoke``/``astream``, ``Command(resume=answer)`` answers a pending ``interrupt()``.
    """

    update: Mapping[str, Any] | None = None
    goto: str | Send | Sequence[str | Send] = field(default=())
    resume: Any = field(default=None)

    @property
    def has_resume(self) -> bool:
        return self.resume is not None


@dataclass(frozen=True, slots=True)
class Interrupt:
    """A question a paused run is waiting on. Answer it by ``id`` when several are pending."""

    value: Any
    id: str
    deadline: datetime | None = None
