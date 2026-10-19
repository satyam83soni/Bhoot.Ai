"""Schema definitions, field specs and field paths.

A ``FieldPath`` is either *concrete* (``transactions[3].balance``) or a *pattern*
(``transactions[*].balance``). Field specs are keyed by pattern; values in a document are
keyed by concrete path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import Any

PathPart = str | int  # int = concrete array index; "*" = any index

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INDEX = re.compile(r"\[(\d+|\*)\]")


class Criticality(StrEnum):
    CRITICAL = "critical"  # amounts, balances, ids: all checks, cross-check allowed
    IMPORTANT = "important"  # dates, names
    NORMAL = "normal"  # descriptions, addresses


class Granularity(StrEnum):
    LEAF = "leaf"  # each scalar has its own value + quote + page
    ROW = "row"  # array items share one quote / page per row (token efficient)


@dataclass(frozen=True, slots=True)
class FieldPath:
    raw: str
    _parts: tuple[PathPart, ...] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_parts", _parse(self.raw))

    @classmethod
    def from_parts(cls, parts: Iterable[PathPart]) -> FieldPath:
        text = ""
        for part in parts:
            if isinstance(part, int) or part == "*":
                text += f"[{part}]"
            else:
                text += f".{part}" if text else part
        return cls(text)

    def parts(self) -> tuple[PathPart, ...]:
        return self._parts

    def is_pattern(self) -> bool:
        return "*" in self._parts

    def matches(self, concrete: FieldPath) -> bool:
        """True if this (pattern or concrete) path matches the concrete path."""
        mine, theirs = self._parts, concrete._parts
        if len(mine) != len(theirs):
            return False
        return all(
            a == b or (a == "*" and isinstance(b, int)) for a, b in zip(mine, theirs, strict=True)
        )

    def pattern(self) -> FieldPath:
        """Concrete → pattern: every index becomes ``*``."""
        return FieldPath.from_parts("*" if isinstance(p, int) else p for p in self._parts)

    def parent_row(self) -> FieldPath | None:
        """``transactions[3].balance`` → ``transactions[3]``; None if not inside an array."""
        for i in range(len(self._parts) - 1, -1, -1):
            if isinstance(self._parts[i], int) or self._parts[i] == "*":
                return FieldPath.from_parts(self._parts[: i + 1])
        return None

    def row_index(self) -> int | None:
        for part in reversed(self._parts):
            if isinstance(part, int):
                return part
        return None

    def with_index(self, index: int) -> FieldPath:
        """Replace the first ``*`` with a concrete index."""
        parts = list(self._parts)
        parts[parts.index("*")] = index
        return FieldPath.from_parts(parts)

    def child(self, name: str) -> FieldPath:
        return FieldPath.from_parts((*self._parts, name))

    @property
    def leaf_name(self) -> str:
        for part in reversed(self._parts):
            if isinstance(part, str) and part != "*":
                return part
        return ""

    def __str__(self) -> str:
        return self.raw


def _parse(raw: str) -> tuple[PathPart, ...]:
    parts: list[PathPart] = []
    pos = 0
    text = raw.strip()
    if not text:
        raise ValueError("empty field path")
    while pos < len(text):
        # a name is allowed at the start or after a dot; an index only after a name or index
        if text[pos] == "[" and parts:
            m = _INDEX.match(text, pos)
            if m:
                parts.append("*" if m.group(1) == "*" else int(m.group(1)))
                pos = m.end()
                continue
        elif not parts or (text[pos] == "." and (pos := pos + 1)):
            m = _NAME.match(text, pos)
            if m:
                parts.append(m.group(0))
                pos = m.end()
                continue
        raise ValueError(f"invalid field path {raw!r} at position {pos}")
    if not isinstance(parts[0], str) or parts[0] == "*":
        raise ValueError(f"field path must start with a name: {raw!r}")
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    path: FieldPath  # pattern path
    json_type: str  # string | number | integer | boolean | date | object | array
    py_type: str  # Decimal | date | str | int | bool | <Enum name>
    criticality: Criticality = Criticality.NORMAL
    granularity: Granularity = Granularity.LEAF
    required: bool = False
    description: str | None = None
    enum: tuple[str, ...] | None = None
    fmt: str | None = None  # date | pan | ifsc | …


@dataclass(frozen=True)
class SchemaDefinition:
    name: str
    version: int  # 0 = ephemeral (a class passed directly, not migrated)
    json_schema: Mapping[str, Any]
    fields: Mapping[str, FieldSpec]  # key = pattern path string
    content_hash: str
    model_import: str | None = None

    def leaf_specs(self) -> list[FieldSpec]:
        return [s for s in self.fields.values() if s.json_type not in ("object", "array")]

    def spec_for(self, concrete: FieldPath) -> FieldSpec:
        spec = self.fields.get(concrete.raw) or self.fields.get(concrete.pattern().raw)
        if spec is None:
            raise KeyError(f"{concrete} is not a field of schema {self.name}@{self.version}")
        return spec

    def critical_paths(self) -> list[FieldSpec]:
        return [s for s in self.leaf_specs() if s.criticality is Criticality.CRITICAL]

    @cached_property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"
