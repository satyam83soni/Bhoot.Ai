"""In-process saver for tests and short-lived runs. Checkpoints are stored serialized, so
what you get back is a copy and anything that would not survive SQLite fails here too."""

from __future__ import annotations

import time

from bhoot_graph.checkpoint.base import BaseCheckpointSaver, Checkpoint, PendingWrite
from bhoot_graph.checkpoint.serde import JsonSerializer


class MemorySaver(BaseCheckpointSaver):
    def __init__(self, serde: JsonSerializer | None = None) -> None:
        super().__init__(serde)
        self._checkpoints: dict[str, list[bytes]] = {}
        self._writes: dict[tuple[str, str], dict[str, bytes]] = {}
        self._locks: dict[str, tuple[str, float]] = {}

    async def aget(self, thread_id: str) -> Checkpoint | None:
        saved = self._checkpoints.get(thread_id)
        return self.load_checkpoint(saved[-1]) if saved else None

    async def alist(self, thread_id: str) -> list[Checkpoint]:
        return [self.load_checkpoint(raw) for raw in self._checkpoints.get(thread_id, [])]

    async def aput(self, checkpoint: Checkpoint) -> None:
        raw = self.dump_checkpoint(checkpoint)  # serialize first: a failure stores nothing
        self._checkpoints.setdefault(checkpoint.thread_id, []).append(raw)

    async def aput_write(self, thread_id: str, checkpoint_id: str, write: PendingWrite) -> None:
        raw = self.dump_write(write)
        self._writes.setdefault((thread_id, checkpoint_id), {})[write.task_id] = raw

    async def aget_writes(self, thread_id: str, checkpoint_id: str) -> dict[str, PendingWrite]:
        saved = self._writes.get((thread_id, checkpoint_id), {})
        return {task_id: self.load_write(raw) for task_id, raw in saved.items()}

    async def adelete_thread(self, thread_id: str) -> None:
        self._checkpoints.pop(thread_id, None)
        self._locks.pop(thread_id, None)
        for key in [k for k in self._writes if k[0] == thread_id]:
            del self._writes[key]

    async def aacquire(self, thread_id: str, owner: str, ttl_s: float) -> bool:
        now = time.monotonic()
        holder = self._locks.get(thread_id)
        if holder is not None and holder[0] != owner and holder[1] > now:
            return False
        self._locks[thread_id] = (owner, now + ttl_s)
        return True

    async def arelease(self, thread_id: str, owner: str) -> None:
        if self._locks.get(thread_id, ("",))[0] == owner:
            del self._locks[thread_id]
