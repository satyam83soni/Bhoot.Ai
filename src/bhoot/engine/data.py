"""Build the nested result data from per-field values keyed by concrete path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bhoot.domain.schema import FieldPath


def assemble(values: Mapping[str, Any]) -> dict[str, Any]:
    """``{"a.b": 1, "rows[1].x": 2}`` -> ``{"a": {"b": 1}, "rows": [{}, {"x": 2}]}``.

    Missing rows in between are filled with empty objects so indices stay aligned.
    """
    root: dict[str, Any] = {}
    for raw in sorted(values, key=_order):
        parts = FieldPath(raw).parts()
        node: Any = root
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            nxt = parts[i + 1] if not last else None
            if isinstance(part, int):
                while len(node) <= part:
                    node.append({} if not isinstance(nxt, int) else [])
                if last:
                    node[part] = values[raw]
                else:
                    node = node[part]
            else:
                if last:
                    node[part] = values[raw]
                else:
                    node = node.setdefault(part, [] if isinstance(nxt, int) else {})
    return root


def _order(raw: str) -> tuple[Any, ...]:
    return tuple((0, p) if isinstance(p, int) else (1, p) for p in FieldPath(raw).parts())
