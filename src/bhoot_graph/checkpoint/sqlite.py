"""SQLite saver (standard library ``sqlite3``). Every write is one transaction, so a crash or a
cancelled run leaves the previous checkpoint intact. WAL mode lets readers run alongside the
writer, and the lock table stops two processes from running the same thread at once."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TypeVar

from bhoot_graph.checkpoint.base import BaseCheckpointSaver, Checkpoint, PendingWrite
from bhoot_graph.checkpoint.serde import JsonSerializer

T = TypeVar("T")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_id, task_id)
);
CREATE TABLE IF NOT EXISTS locks (
    thread_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    expires_at REAL NOT NULL
);
"""


class SqliteSaver(BaseCheckpointSaver):
    def __init__(self, path: str | Path = ":memory:", serde: JsonSerializer | None = None) -> None:
        super().__init__(serde)
        self.path = str(path)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._guard = threading.Lock()
        with self._guard:
            self._db.execute("PRAGMA busy_timeout = 10000")
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(_SCHEMA)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        with self._guard:
            self._db.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ plumbing
    async def _run(self, work: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``work`` in one transaction on a worker thread. If the awaiting task is
        cancelled, the thread still commits or rolls back the whole transaction."""

        def transaction() -> T:
            with self._guard:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    result = work(self._db)
                except BaseException:
                    self._db.execute("ROLLBACK")
                    raise
                self._db.execute("COMMIT")
                return result

        return await asyncio.to_thread(transaction)

    # ------------------------------------------------------------------ checkpoints
    async def aget(self, thread_id: str) -> Checkpoint | None:
        row = await self._run(
            lambda db: db.execute(
                "SELECT data FROM checkpoints WHERE thread_id = ? "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        )
        return self.load_checkpoint(row[0]) if row else None

    async def alist(self, thread_id: str) -> list[Checkpoint]:
        rows = await self._run(
            lambda db: db.execute(
                "SELECT data FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id",
                (thread_id,),
            ).fetchall()
        )
        return [self.load_checkpoint(row[0]) for row in rows]

    async def aput(self, checkpoint: Checkpoint) -> None:
        raw = self.dump_checkpoint(checkpoint)
        await self._run(
            lambda db: db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
                (checkpoint.thread_id, checkpoint.id, raw),
            )
        )

    async def aput_write(self, thread_id: str, checkpoint_id: str, write: PendingWrite) -> None:
        raw = self.dump_write(write)
        await self._run(
            lambda db: db.execute(
                "INSERT OR REPLACE INTO writes VALUES (?, ?, ?, ?)",
                (thread_id, checkpoint_id, write.task_id, raw),
            )
        )

    async def aget_writes(self, thread_id: str, checkpoint_id: str) -> dict[str, PendingWrite]:
        rows = await self._run(
            lambda db: db.execute(
                "SELECT task_id, data FROM writes WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_id),
            ).fetchall()
        )
        return {task_id: self.load_write(data) for task_id, data in rows}

    async def adelete_thread(self, thread_id: str) -> None:
        def delete(db: sqlite3.Connection) -> None:
            for table in ("checkpoints", "writes", "locks"):
                db.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))

        await self._run(delete)

    # ------------------------------------------------------------------ locks
    async def aacquire(self, thread_id: str, owner: str, ttl_s: float) -> bool:
        def acquire(db: sqlite3.Connection) -> bool:
            now = time.time()  # wall clock: the lock is shared between processes
            row: Any = db.execute(
                "SELECT owner, expires_at FROM locks WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if row is not None and row[0] != owner and row[1] > now:
                return False
            db.execute(
                "INSERT OR REPLACE INTO locks VALUES (?, ?, ?)", (thread_id, owner, now + ttl_s)
            )
            return True

        return await self._run(acquire)

    async def arelease(self, thread_id: str, owner: str) -> None:
        await self._run(
            lambda db: db.execute(
                "DELETE FROM locks WHERE thread_id = ? AND owner = ?", (thread_id, owner)
            )
        )
