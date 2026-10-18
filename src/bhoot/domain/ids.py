"""Sortable identifiers for jobs and runs (ULID-like: time prefix + randomness).

The time and randomness are parameters so callers (and tests) control them; the defaults use
the system clock and ``secrets``.
"""

from __future__ import annotations

import secrets
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        value, rem = divmod(value, 32)
        chars.append(_ALPHABET[rem])
    return "".join(reversed(chars))


def new_id(prefix: str, *, timestamp_ms: int | None = None, randomness: bytes | None = None) -> str:
    ts = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
    rand = secrets.token_bytes(10) if randomness is None else randomness
    if len(rand) != 10:
        raise ValueError("randomness must be 10 bytes")
    return f"{prefix}_{_encode(ts, 10)}{_encode(int.from_bytes(rand, 'big'), 16)}"


def new_job_id(*, timestamp_ms: int | None = None, randomness: bytes | None = None) -> str:
    return new_id("job", timestamp_ms=timestamp_ms, randomness=randomness)


def new_run_id(*, timestamp_ms: int | None = None, randomness: bytes | None = None) -> str:
    return new_id("run", timestamp_ms=timestamp_ms, randomness=randomness)
