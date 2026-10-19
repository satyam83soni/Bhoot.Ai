"""Versioned JSON serializer for checkpoints.

Plain JSON values pass through; ``Decimal``, dates and times, ``UUID``, bytes, tuples, sets,
non-string dict keys and ``Send`` are tagged so they come back as the same type. Your own
classes (dataclasses, enums, models) must be registered: a checkpoint never names a class that
is then imported, so loading one cannot run arbitrary code.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import json
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from bhoot_graph.errors import GraphError
from bhoot_graph.types import Send

FORMAT = "bhoot_graph.json"
VERSION = 1
_TAG = "__t"


class SerializationError(GraphError):
    """A value cannot be written to or read from a checkpoint."""


class JsonSerializer:
    def __init__(self) -> None:
        self._encoders: dict[type, tuple[str, Callable[[Any], Any]]] = {}
        self._decoders: dict[str, Callable[[Any], Any]] = {}

    # ------------------------------------------------------------------ registry
    def register(
        self,
        cls: type,
        *,
        encode: Callable[[Any], Any],
        decode: Callable[[Any], Any],
        name: str | None = None,
    ) -> None:
        """Teach the serializer a type. ``encode`` returns JSON-able data (it is encoded
        recursively), ``decode`` rebuilds the object from it. ``name`` is what checkpoints
        store; keep it stable across releases."""
        tag = name or f"{cls.__module__}.{cls.__qualname__}"
        if tag in self._decoders and self._encoders.get(cls, (None,))[0] != tag:
            raise SerializationError(f"type name {tag!r} is already registered")
        self._encoders[cls] = (tag, encode)
        self._decoders[tag] = decode

    def register_dataclass(self, cls: type, *, name: str | None = None) -> None:
        if not dataclasses.is_dataclass(cls):
            raise SerializationError(f"{cls!r} is not a dataclass")
        init_fields = [f.name for f in dataclasses.fields(cls) if f.init]
        self.register(
            cls,
            encode=lambda obj: {key: getattr(obj, key) for key in init_fields},
            decode=lambda data: cls(**data),
            name=name,
        )

    def register_enum(self, cls: type[enum.Enum], *, name: str | None = None) -> None:
        self.register(cls, encode=lambda member: member.value, decode=cls, name=name)

    # ------------------------------------------------------------------ bytes
    def dumps(self, value: Any) -> bytes:
        document = {"format": FORMAT, "version": VERSION, "data": self.encode(value)}
        return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def loads(self, raw: bytes) -> Any:
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SerializationError(f"checkpoint is not valid JSON: {exc}") from exc
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            raise SerializationError("not a bhoot_graph checkpoint")
        if document.get("version") != VERSION:
            raise SerializationError(
                f"checkpoint format version {document.get('version')!r} is not supported "
                f"(this release reads version {VERSION})"
            )
        return self.decode(document["data"])

    # ------------------------------------------------------------------ values
    def encode(self, value: Any) -> Any:
        registered = self._encoders.get(type(value))
        if registered is not None:
            tag, encode = registered
            return {_TAG: "obj", "type": tag, "v": self.encode(encode(value))}
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, dict):
            if _TAG not in value and all(isinstance(key, str) for key in value):
                return {key: self.encode(item) for key, item in value.items()}
            return {_TAG: "dict", "v": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, tuple):
            return {_TAG: "tuple", "v": [self.encode(item) for item in value]}
        if isinstance(value, set | frozenset):
            kind = "frozenset" if isinstance(value, frozenset) else "set"
            return {_TAG: kind, "v": [self.encode(item) for item in value]}
        if isinstance(value, Decimal):
            return {_TAG: "decimal", "v": str(value)}
        if isinstance(value, datetime):  # before date: a datetime is a date
            return {_TAG: "datetime", "v": value.isoformat()}
        if isinstance(value, date):
            return {_TAG: "date", "v": value.isoformat()}
        if isinstance(value, time):
            return {_TAG: "time", "v": value.isoformat()}
        if isinstance(value, timedelta):
            return {_TAG: "timedelta", "v": [value.days, value.seconds, value.microseconds]}
        if isinstance(value, UUID):
            return {_TAG: "uuid", "v": str(value)}
        if isinstance(value, bytes):
            return {_TAG: "bytes", "v": base64.b64encode(value).decode("ascii")}
        if isinstance(value, Send):
            return {_TAG: "send", "node": value.node, "v": self.encode(value.arg)}
        cls = type(value)
        raise SerializationError(
            f"cannot checkpoint a {cls.__module__}.{cls.__qualname__}; register it, e.g. "
            "serde.register_dataclass(...), serde.register_enum(...) or serde.register(...)"
        )

    def decode(self, data: Any) -> Any:
        if isinstance(data, list):
            return [self.decode(item) for item in data]
        if not isinstance(data, dict):
            return data
        tag = data.get(_TAG)
        if tag is None:
            return {key: self.decode(item) for key, item in data.items()}
        raw: Any = data.get("v")
        if tag == "obj":
            decode = self._decoders.get(data["type"])
            if decode is None:
                raise SerializationError(
                    f"checkpoint holds a {data['type']!r}, which is not registered here"
                )
            return decode(self.decode(raw))
        if tag == "dict":
            return {self.decode(k): self.decode(v) for k, v in raw}
        if tag == "tuple":
            return tuple(self.decode(item) for item in raw)
        if tag == "set":
            return {self.decode(item) for item in raw}
        if tag == "frozenset":
            return frozenset(self.decode(item) for item in raw)
        if tag == "send":
            return Send(data["node"], self.decode(raw))
        simple: dict[str, Callable[[Any], Any]] = {
            "decimal": Decimal,
            "datetime": datetime.fromisoformat,
            "date": date.fromisoformat,
            "time": time.fromisoformat,
            "timedelta": lambda v: timedelta(days=v[0], seconds=v[1], microseconds=v[2]),
            "uuid": UUID,
            "bytes": base64.b64decode,
        }
        if tag not in simple:
            raise SerializationError(f"unknown checkpoint tag {tag!r}")
        return simple[tag](raw)
