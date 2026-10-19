"""Turn a pydantic schema class into a ``SchemaDefinition``: field specs keyed by pattern path,
plus a canonical JSON Schema whose hash identifies the schema version."""

from __future__ import annotations

import datetime as dt
import enum
import types
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from bhoot.domain.hashing import content_hash
from bhoot.domain.schema import Criticality, FieldPath, FieldSpec, Granularity, SchemaDefinition

_SCALARS: dict[type, tuple[str, str]] = {  # python type -> (json_type, py_type)
    Decimal: ("number", "Decimal"),
    float: ("number", "Decimal"),  # money is never float: read as Decimal, converted at the end
    int: ("integer", "int"),
    bool: ("boolean", "bool"),
    str: ("string", "str"),
    dt.date: ("date", "date"),
}


class SchemaIntrospectionError(TypeError):
    """The model uses a type bhoot cannot extract (say which field and what to use)."""


def introspect(
    model: type[BaseModel], *, name: str | None = None, version: int = 0
) -> SchemaDefinition:
    fields: dict[str, FieldSpec] = {}
    _walk_model(model, (), fields, inherited_granularity=None)
    json_schema = _inline_refs(model.model_json_schema())
    return SchemaDefinition(
        name=name or _snake(model.__name__),
        version=version,
        json_schema=json_schema,
        fields=fields,
        content_hash=content_hash({"schema": json_schema, "fields": _spec_summary(fields)}),
        model_import=f"{model.__module__}:{model.__qualname__}",
    )


def _walk_model(
    model: type[BaseModel],
    prefix: tuple[str | int, ...],
    out: dict[str, FieldSpec],
    inherited_granularity: Granularity | None,
) -> None:
    for key, info in model.model_fields.items():
        name = info.alias or key
        _walk_field((*prefix, name), info.annotation, info, out, inherited_granularity)


def _walk_field(
    parts: tuple[str | int, ...],
    annotation: Any,
    info: FieldInfo,
    out: dict[str, FieldSpec],
    inherited_granularity: Granularity | None,
) -> None:
    path = FieldPath.from_parts(parts)
    inner, optional = _unwrap_optional(annotation)
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
    criticality = Criticality(str(extra.get("x-criticality", Criticality.NORMAL.value)))
    granularity = Granularity(str(extra["x-granularity"])) if "x-granularity" in extra else None
    required = not optional and info.is_required()
    fmt = str(extra["x-format"]) if "x-format" in extra else None

    def spec(json_type: str, py_type: str, granularity: Granularity, **more: Any) -> FieldSpec:
        return FieldSpec(
            path,
            json_type,
            py_type,
            criticality=criticality,
            granularity=granularity,
            required=required,
            description=info.description,
            **more,
        )

    if get_origin(inner) in (list, tuple):
        args = [a for a in get_args(inner) if a is not Ellipsis]
        if len(args) != 1:
            raise SchemaIntrospectionError(f"{path}: use list[X] for repeated values")
        item, _ = _unwrap_optional(args[0])
        rows = isinstance(item, type) and issubclass(item, BaseModel)
        row_granularity = granularity or (Granularity.ROW if rows else Granularity.LEAF)
        out[path.raw] = spec("array", "list", row_granularity)
        child = (*parts, "*")
        if rows:
            _walk_model(item, child, out, row_granularity)
        else:
            _walk_field(child, item, FieldInfo(annotation=item), out, row_granularity)
        return

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        out[path.raw] = spec("object", inner.__name__, inherited_granularity or Granularity.LEAF)
        _walk_model(inner, parts, out, inherited_granularity)
        return

    json_type, py_type, enum_values = _scalar(path, inner)
    out[path.raw] = spec(
        json_type,
        py_type,
        inherited_granularity or Granularity.LEAF,
        enum=enum_values,
        fmt=fmt or ("date" if json_type == "date" else None),
    )


def _scalar(path: FieldPath, tp: Any) -> tuple[str, str, tuple[str, ...] | None]:
    if get_origin(tp) is Literal:
        values = tuple(str(v) for v in get_args(tp))
        return "string", "str", values
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return "string", tp.__name__, tuple(str(m.value) for m in tp)
    if tp is dt.datetime:
        raise SchemaIntrospectionError(f"{path}: use date (documents print dates, not instants)")
    for py, (json_type, py_type) in _SCALARS.items():
        if tp is py:
            return json_type, py_type, None
    raise SchemaIntrospectionError(
        f"{path}: unsupported type {tp!r}; use str, Decimal, int, bool, date, an Enum, "
        "Literal, a nested Schema or list[...]"
    )


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) != 1:
            raise SchemaIntrospectionError(f"unions other than X | None are not supported: {tp!r}")
        return args[0], len(args) < len(get_args(tp))
    return tp, False


def _inline_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = resolve(defs[ref.removeprefix("#/$defs/")])
                return {**target, **{k: resolve(v) for k, v in node.items() if k != "$ref"}}
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(dict(schema))  # type: ignore[no-any-return]


def _spec_summary(fields: Mapping[str, FieldSpec]) -> list[list[Any]]:
    return [
        [
            k,
            s.json_type,
            s.py_type,
            s.criticality.value,
            s.granularity.value,
            s.required,
            s.enum,
            s.fmt,
        ]
        for k, s in fields.items()
    ]


def _snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
