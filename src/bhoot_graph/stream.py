"""Streaming: the events of one ``astream`` run and the writer nodes use to add their own.

Modes:

* ``values``: the whole state, after the input and after every superstep;
* ``updates``: ``{node: update}`` for every task, in task order;
* ``custom``: whatever a node passes to ``get_stream_writer()``;
* ``messages``: ``(chunk, {"node", "task_id"})`` from ``get_stream_writer("messages")``,
  e.g. model tokens;
* ``tasks``: task start and result events with timings and errors;
* ``debug``: the task events plus one event per saved checkpoint.

``values`` and ``updates`` come in a fixed order. ``custom`` and ``messages`` arrive as they
are written, so parallel branches interleave.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Any

from bhoot_graph.errors import GraphError

MODES = frozenset({"values", "updates", "custom", "messages", "tasks", "debug"})
StreamWriter = Callable[[Any], None]


def _discard(chunk: Any) -> None:
    return None


_WRITERS: ContextVar[dict[str, StreamWriter] | None] = ContextVar(
    "bhoot_graph_writers", default=None
)


def get_stream_writer(mode: str = "custom") -> StreamWriter:
    """Inside a node: a function that sends a chunk to the ``custom`` (or ``messages``)
    stream. Outside ``astream``, or when nobody asked for that mode, writes are dropped."""
    if mode not in ("custom", "messages"):
        raise GraphError(f"nodes can write to 'custom' or 'messages', not {mode!r}")
    writers = _WRITERS.get()
    return writers.get(mode, _discard) if writers else _discard


def parse_modes(stream_mode: str | Sequence[str]) -> tuple[frozenset[str], bool]:
    """Returns the modes and whether a single mode was asked for (then chunks come bare,
    otherwise as ``(mode, chunk)`` pairs, as in LangGraph)."""
    single = isinstance(stream_mode, str)
    modes = frozenset([stream_mode] if isinstance(stream_mode, str) else stream_mode)
    unknown = modes - MODES
    if not modes or unknown:
        raise GraphError(f"unknown stream mode(s) {sorted(unknown)}; choose from {sorted(MODES)}")
    return modes, single


class RunStream:
    """The event queue of one run.

    The run loop awaits ``push``, so a slow consumer pauses the graph once ``maxsize``
    events are waiting. Nodes call ``push_nowait`` (from any thread) and never block;
    their events wait in an overflow buffer that the loop drains, in order, before it
    pushes anything else.
    """

    def __init__(self, modes: frozenset[str], maxsize: int) -> None:
        self.modes = modes
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(1, maxsize))
        self._overflow: deque[tuple[str, Any]] = deque()
        self._loop = asyncio.get_running_loop()
        self._thread = threading.get_ident()

    def wants(self, mode: str) -> bool:
        return mode in self.modes

    async def push(self, mode: str, chunk: Any) -> None:
        if mode in self.modes:
            await self.flush()
            await self.queue.put((mode, chunk))

    def push_nowait(self, mode: str, chunk: Any) -> None:
        if mode not in self.modes:
            return
        if threading.get_ident() != self._thread:
            self._loop.call_soon_threadsafe(self._buffer, mode, chunk)
        else:
            self._buffer(mode, chunk)

    def _buffer(self, mode: str, chunk: Any) -> None:
        if self._overflow or self.queue.full():
            self._overflow.append((mode, chunk))
        else:
            self.queue.put_nowait((mode, chunk))

    async def flush(self) -> None:
        while self._overflow:
            await self.queue.put(self._overflow.popleft())

    async def close(self, end: Any) -> None:
        """Deliver everything still buffered, then ``end`` (a done marker or an error)."""
        await self.flush()
        await self.queue.put(end)

    def task_event(self, chunk: dict[str, Any]) -> None:
        self.push_nowait("tasks", chunk)
        self.push_nowait("debug", chunk)

    def writers_for(self, node: str, task_id: str) -> dict[str, StreamWriter]:
        def custom(chunk: Any) -> None:
            self.push_nowait("custom", chunk)

        def messages(chunk: Any) -> None:
            self.push_nowait("messages", (chunk, {"node": node, "task_id": task_id}))

        return {"custom": custom, "messages": messages}


def set_writers(writers: dict[str, StreamWriter]) -> None:
    """Called at the start of a task: the task runs in its own context, so this stays local
    to it and to whatever it starts (threads, subgraphs)."""
    _WRITERS.set(writers)
