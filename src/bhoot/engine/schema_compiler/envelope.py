"""Evidence envelopes: the JSON Schema we actually ask models to fill.

Every value comes back as the text printed on the page plus the proof for it:

* leaf fields: ``{"value": "1,25,000.00", "quote": "Closing Balance 1,25,000.00", "page": 3}``;
* row arrays (transactions): each row has its columns plus one ``_quote`` (the full row text)
  and ``_page``, about three times fewer output tokens than a quote per cell.

Values are always strings "exactly as printed": the engine parses them, so a model can't
silently reformat a number or a date. Every property is required and nullable and no extra
keys are allowed, which strict structured-output modes need.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bhoot.domain.schema import FieldPath, FieldSpec, SchemaDefinition
from bhoot.engine.schema_compiler.tree import Parts, SchemaTree, Selection

NULLABLE_STRING = ["string", "null"]
QUOTE_HINT = "The shortest verbatim text from the page that contains the value; null if absent."
PAGE_HINT = "Page number (1-based) the quote is on; null if absent."
ROW_QUOTE_HINT = "The full text of this row exactly as printed on the page."
ROW_PAGE_HINT = "Page number (1-based) this row is on."


def value_hint(spec: FieldSpec) -> str:
    kind = {
        "number": "Amount or number exactly as printed (keep separators, currency, Dr/Cr).",
        "integer": "Number exactly as printed.",
        "date": "Date exactly as printed (do not reformat).",
        "boolean": "Exactly as printed (e.g. Yes/No).",
    }.get(spec.json_type, "Text exactly as printed.")
    parts = [spec.description, kind] if spec.description else [kind]
    if spec.enum:
        parts.append("One of: " + ", ".join(spec.enum) + " (or the printed label).")
    parts.append("null if it is not in the document; never guess.")
    return " ".join(p.rstrip(".") + "." for p in parts)


def _obj(properties: dict[str, Any], description: str | None = None) -> dict[str, Any]:
    node: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if description:
        node["description"] = description
    return node


class EvidenceSchemaBuilder:
    """Builds the envelope for a whole schema, or only for some fields (``only=``) when
    repairing. Paths may be concrete or patterns; rows are always asked for whole."""

    def build(
        self, schema: SchemaDefinition, only: Iterable[FieldPath] | None = None
    ) -> dict[str, Any]:
        tree = SchemaTree(schema)
        selection = Selection(tree, only)
        properties = {
            spec.path.leaf_name: self._node(tree, spec, selection)
            for spec in tree.roots()
            if selection.keeps(spec.path.parts())
        }
        if not properties:
            raise ValueError(f"no fields of {schema.name} match {list(only or [])}")
        return _obj(properties)

    def _node(self, tree: SchemaTree, spec: FieldSpec, selection: Selection) -> dict[str, Any]:
        parts = spec.path.parts()
        if spec.json_type == "object":
            return _obj(self._children(tree, parts, selection), spec.description)
        if spec.json_type == "array":
            return {"type": "array", "items": self._item(tree, spec, selection)} | (
                {"description": spec.description} if spec.description else {}
            )
        return self._leaf(spec)

    def _children(self, tree: SchemaTree, parts: Parts, selection: Selection) -> dict[str, Any]:
        return {
            child.path.leaf_name: self._node(tree, child, selection)
            for child in tree.children(parts)
            if selection.keeps(child.path.parts())
        }

    def _item(self, tree: SchemaTree, array: FieldSpec, selection: Selection) -> dict[str, Any]:
        item_parts = (*array.path.parts(), "*")
        if tree.is_rows(array):
            columns = {
                child.path.leaf_name: self._row_column(tree, child)
                for child in tree.children(item_parts)
            }
            columns["_quote"] = {"type": NULLABLE_STRING, "description": ROW_QUOTE_HINT}
            columns["_page"] = {"type": ["integer", "null"], "description": ROW_PAGE_HINT}
            return _obj(columns)
        scalar = tree.item_spec(array)
        if scalar is not None:
            return self._leaf(scalar)
        return _obj(self._children(tree, item_parts, selection))

    def _row_column(self, tree: SchemaTree, spec: FieldSpec) -> dict[str, Any]:
        if spec.json_type == "object":
            columns = {
                c.path.leaf_name: self._row_column(tree, c)
                for c in tree.children(spec.path.parts())
            }
            return _obj(columns, spec.description)
        if spec.json_type == "array":
            raise ValueError(f"{spec.path}: lists inside table rows are not supported")
        return {"type": NULLABLE_STRING, "description": value_hint(spec)}

    @staticmethod
    def _leaf(spec: FieldSpec) -> dict[str, Any]:
        return _obj(
            {
                "value": {"type": NULLABLE_STRING, "description": value_hint(spec)},
                "quote": {"type": NULLABLE_STRING, "description": QUOTE_HINT},
                "page": {"type": ["integer", "null"], "description": PAGE_HINT},
            }
        )
