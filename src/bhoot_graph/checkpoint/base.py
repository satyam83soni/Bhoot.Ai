"""What a checkpoint holds and what a saver must do with it.

A checkpoint is written after every superstep: the state, the tasks scheduled next and the
progress of waiting edges. When a step fails, the tasks that did finish are kept as pending
writes, so resuming runs only the work that is still missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from bhoot_graph.checkpoint.serde import JsonSerializer
from bhoot_graph.errors import GraphError


class CheckpointMismatchError(GraphError):
    """The saved run belongs to a different graph or graph version."""


class ThreadBusyError(GraphError):
    """Another run holds the lock on this thread."""


@dataclass(frozen=True, slots=True)
class Task:
    id: str  # "<step>:<index>:<node>", zero-padded so ids sort in scheduling order
    node: str
    arg: Any = None
    sent: bool = False  # True: ``arg`` is a Send payload, not the state


@dataclass(frozen=True, slots=True)
class PendingWrite:
    """The result of a task that finished in a step that failed overall."""

    task_id: str
    update: Mapping[str, Any]
    goto: tuple[Any, ...] = ()  # node names and Sends


@dataclass(frozen=True, slots=True)
class Checkpoint:
    thread_id: str
    step: int
    values: Mapping[str, Any]
    next: tuple[Task, ...]
    waiting: Mapping[int, tuple[str, ...]]  # waiting-edge index -> sources already finished
    graph: str
    version: str | None
    fingerprint: str  # hash of the graph's nodes and edges
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def id(self) -> str:
        return f"{self.step:010d}"

    @property
    def done(self) -> bool:
        return not self.next


class BaseCheckpointSaver(ABC):
    """Storage for checkpoints. Every write must be atomic: a crash or cancellation leaves
    either the old checkpoint or the new one, never half of one."""

    def __init__(self, serde: JsonSerializer | None = None) -> None:
        self.serde = serde or JsonSerializer()

    @abstractmethod
    async def aget(self, thread_id: str) -> Checkpoint | None:
        """The latest checkpoint of a thread."""

    @abstractmethod
    async def alist(self, thread_id: str) -> list[Checkpoint]:
        """All checkpoints of a thread, oldest first."""

    @abstractmethod
    async def aput(self, checkpoint: Checkpoint) -> None: ...

    @abstractmethod
    async def aput_write(self, thread_id: str, checkpoint_id: str, write: PendingWrite) -> None:
        """Keep a finished task's result from a step that failed."""

    @abstractmethod
    async def aget_writes(self, thread_id: str, checkpoint_id: str) -> dict[str, PendingWrite]:
        """Pending writes made on top of a checkpoint, by task id."""

    @abstractmethod
    async def adelete_thread(self, thread_id: str) -> None: ...

    @abstractmethod
    async def aacquire(self, thread_id: str, owner: str, ttl_s: float) -> bool:
        """Take or extend the lock on a thread. False if another owner holds a live lock."""

    @abstractmethod
    async def arelease(self, thread_id: str, owner: str) -> None: ...

    # ------------------------------------------------------------------ encoding helpers
    def dump_checkpoint(self, checkpoint: Checkpoint) -> bytes:
        return self.serde.dumps(
            {
                "thread_id": checkpoint.thread_id,
                "step": checkpoint.step,
                "values": dict(checkpoint.values),
                "next": [[t.id, t.node, t.arg, t.sent] for t in checkpoint.next],
                "waiting": {str(k): list(v) for k, v in checkpoint.waiting.items()},
                "graph": checkpoint.graph,
                "version": checkpoint.version,
                "fingerprint": checkpoint.fingerprint,
                "created_at": checkpoint.created_at,
            }
        )

    def load_checkpoint(self, raw: bytes) -> Checkpoint:
        data = self.serde.loads(raw)
        return Checkpoint(
            thread_id=data["thread_id"],
            step=data["step"],
            values=data["values"],
            next=tuple(Task(*item) for item in data["next"]),
            waiting={int(k): tuple(v) for k, v in data["waiting"].items()},
            graph=data["graph"],
            version=data["version"],
            fingerprint=data["fingerprint"],
            created_at=data["created_at"],
        )

    def dump_write(self, write: PendingWrite) -> bytes:
        return self.serde.dumps([write.task_id, dict(write.update), list(write.goto)])

    def load_write(self, raw: bytes) -> PendingWrite:
        task_id, update, goto = self.serde.loads(raw)
        return PendingWrite(task_id, update, tuple(goto))
