"""Blob stores for page images and crops. Keys are content addressed
(``<namespace>/<sha256><suffix>``), so the same image is stored once and equal images give
equal LLM cache keys."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path

_NAMESPACE = re.compile(r"^[a-z0-9_-]{1,40}$")
_SUFFIX = re.compile(r"^(\.[a-z0-9]{1,8})?$")
_KEY = re.compile(r"^[a-z0-9_-]{1,40}/[0-9a-f]{64}(\.[a-z0-9]{1,8})?$")


def blob_key(data: bytes, namespace: str, suffix: str) -> str:
    if not _NAMESPACE.match(namespace) or not _SUFFIX.match(suffix):
        raise ValueError(f"invalid blob namespace {namespace!r} or suffix {suffix!r}")
    return f"{namespace}/{hashlib.sha256(data).hexdigest()}{suffix}"


def _check_key(key: str) -> str:
    if not _KEY.match(key):
        raise ValueError(f"invalid blob key {key!r}")
    return key


class MemoryBlobStore:
    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    async def put(self, data: bytes, *, namespace: str, suffix: str) -> str:
        key = blob_key(data, namespace, suffix)
        self._blobs[key] = bytes(data)
        return key

    async def get(self, key: str) -> bytes:
        try:
            return self._blobs[_check_key(key)]
        except KeyError:
            raise KeyError(f"no blob {key!r}") from None

    async def exists(self, key: str) -> bool:
        return _check_key(key) in self._blobs


class FileBlobStore:
    """Blobs as files under ``root``; writes go to a temp file first, then an atomic rename."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / _check_key(key)

    async def put(self, data: bytes, *, namespace: str, suffix: str) -> str:
        key = blob_key(data, namespace, suffix)
        path = self._path(key)
        if not path.exists():
            await asyncio.to_thread(_atomic_write, path, bytes(data))
        return key

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError:
            raise KeyError(f"no blob {key!r}") from None

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
