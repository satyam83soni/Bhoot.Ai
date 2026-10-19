"""Human in the loop: a node pauses the run with ``interrupt(question)`` and gets the answer
when the run is resumed with ``ainvoke(Command(resume=answer), config)``.

On resume the node runs again from the top; this time ``interrupt()`` returns the answer
instead of pausing. A node may ask several questions; answers come back in order. Work
done before ``interrupt()`` therefore runs again, so keep it cheap or cached.

Pausing needs a checkpointer: the answer can arrive minutes or days later, in another
process.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from bhoot_graph.errors import GraphError
from bhoot_graph.types import Interrupt

_NO_DEFAULT: Any = object()


def _now() -> datetime:
    return datetime.now(UTC)


class GraphInterrupt(BaseException):  # a signal, like KeyboardInterrupt
    """Raised by ``interrupt()`` to pause the run.

    It derives from BaseException so an ``except Exception`` in a node does not swallow it.
    """

    def __init__(self, interrupt: Interrupt) -> None:
        super().__init__(interrupt.value)
        self.interrupt = interrupt


@dataclass(slots=True)
class _TaskScratch:
    task_id: str
    resumes: tuple[Any, ...]
    calls: int = field(default=0)


_SCRATCH: ContextVar[_TaskScratch | None] = ContextVar("bhoot_graph_task", default=None)


def start_task(task_id: str, resumes: tuple[Any, ...]) -> None:
    """Called by the runner at the start of each task (in the task's own context)."""
    _SCRATCH.set(_TaskScratch(task_id, resumes))


def interrupt(value: Any, *, deadline: datetime | None = None, default: Any = _NO_DEFAULT) -> Any:
    """Pause the run and show ``value`` to whoever answers. Returns the answer on resume.

    With a ``deadline`` (timezone-aware), resuming after it without an answer returns
    ``default`` instead of pausing again.
    """
    scratch = _SCRATCH.get()
    if scratch is None:
        raise GraphError("interrupt() can only be called inside a graph node")
    if deadline is not None:
        if deadline.tzinfo is None:
            raise GraphError("interrupt(deadline=...) needs a timezone-aware datetime")
        if default is _NO_DEFAULT:
            raise GraphError("interrupt(deadline=...) needs a default answer for when it passes")
    index = scratch.calls
    scratch.calls += 1
    if index < len(scratch.resumes):
        return scratch.resumes[index]
    if deadline is not None and _now() >= deadline:
        return default
    raise GraphInterrupt(Interrupt(value, f"{scratch.task_id}#{index}", deadline))
