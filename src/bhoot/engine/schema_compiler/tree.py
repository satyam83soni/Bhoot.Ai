"""Walk a ``SchemaDefinition`` as a tree of field specs, optionally limited to some paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bhoot.domain.schema import FieldPath, FieldSpec, Granularity, PathPart, SchemaDefinition

Parts = tuple[PathPart, ...]
STAR = "*"


@dataclass(frozen=True)
class SchemaTree:
    schema: SchemaDefinition

    def roots(self) -> list[FieldSpec]:
        return [s for s in self.schema.fields.values() if len(s.path.parts()) == 1]

    def children(self, parts: Parts) -> list[FieldSpec]:
        """Specs one level below ``parts`` (for an array, below its ``[*]`` item)."""
        n = len(parts)
        return [
            s
            for s in self.schema.fields.values()
            if len(s.path.parts()) == n + 1 and s.path.parts()[:n] == parts
        ]

    def item_spec(self, array: FieldSpec) -> FieldSpec | None:
        """The spec of a scalar list's items (``tags[*]``); None for lists of rows."""
        return self.schema.fields.get(FieldPath.from_parts((*array.path.parts(), STAR)).raw)

    @staticmethod
    def is_rows(array: FieldSpec) -> bool:
        return array.json_type == "array" and array.granularity is Granularity.ROW


class Selection:
    """Which parts of the tree to include. A field is kept when it, one of its ancestors or one
    of its descendants was asked for; inside a row array the whole row is kept, because a row
    is quoted and checked as one unit."""

    def __init__(self, tree: SchemaTree, only: Iterable[FieldPath] | None) -> None:
        self._all = only is None
        self._wanted: list[Parts] = []
        for path in only or ():
            parts = path.pattern().parts()
            self._wanted.append(_row_root(tree, parts) or parts)

    def keeps(self, parts: Parts) -> bool:
        if self._all:
            return True
        for wanted in self._wanted:
            n = min(len(wanted), len(parts))
            if wanted[:n] == parts[:n]:
                return True
        return False


def _row_root(tree: SchemaTree, parts: Parts) -> Parts | None:
    """For a path inside a row array, the array's own parts."""
    for i in range(1, len(parts) + 1):
        spec = tree.schema.fields.get(FieldPath.from_parts(parts[:i]).raw)
        if spec is not None and SchemaTree.is_rows(spec):
            return parts[:i]
    return None
