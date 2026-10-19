"""What can be passed as a document, and how it is read."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from bhoot.domain.errors import DocumentError, FileTooLargeError

InputLike = str | Path | bytes | bytearray | BinaryIO


def read_input(source: InputLike, *, max_mb: int) -> tuple[bytes, str | None]:
    """The document's bytes and file name (if it has one). Paths are read, files are read from
    their current position, bytes are used as they are."""
    name: str | None = None
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DocumentError(f"no file at {path}")
        size = path.stat().st_size
        if size > max_mb * 1024 * 1024:
            raise FileTooLargeError(f"{path.name} is {size / 1e6:.1f} MB; the limit is {max_mb} MB")
        data, name = path.read_bytes(), path.name
    elif hasattr(source, "read"):
        data = source.read()
        if not isinstance(data, bytes):
            raise DocumentError("open the file in binary mode ('rb')")
        name = (
            getattr(source, "name", None)
            if isinstance(getattr(source, "name", None), str)
            else None
        )
    else:
        raise TypeError(
            f"cannot read a document from {type(source).__name__}; pass a path, bytes or a binary file"
        )
    if len(data) > max_mb * 1024 * 1024:
        raise FileTooLargeError(f"document is {len(data) / 1e6:.1f} MB; the limit is {max_mb} MB")
    if not data:
        raise DocumentError("the document is empty")
    return data, Path(name).name if name else None
