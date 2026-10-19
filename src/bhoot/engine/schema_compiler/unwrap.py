"""Read a model's envelope answer back into candidate values.

Values stay as the printed strings (normalisation happens later). Leaf fields answered with
null become ``nulls`` (the policy decides between "absent" and "ask again"), never candidates.
Rows get indices in answer order, offset by the item's row window when repairing a window;
the merge step assigns final indices across pages.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bhoot.domain.candidate import CandidateRole, CandidateValue
from bhoot.domain.errors import ResponseParseError
from bhoot.domain.schema import FieldPath, FieldSpec, SchemaDefinition
from bhoot.domain.work import WorkItem, WorkKind
from bhoot.engine.schema_compiler.tree import Parts, SchemaTree, Selection
from bhoot.ports.llm import LLMResponse

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_answer(response: LLMResponse) -> Mapping[str, Any]:
    """The answer as a JSON object, tolerating code fences and prose around it."""
    if isinstance(response.parsed, Mapping):
        return response.parsed
    text = _FENCE.sub("", (response.content or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
    raise ResponseParseError(f"{response.model}: the answer is not a JSON object")


@dataclass(frozen=True, slots=True)
class Unwrapped:
    candidates: tuple[CandidateValue, ...]
    nulls: tuple[FieldPath, ...]  # leaf fields the model said are not in the document
    row_counts: Mapping[str, int]  # row array pattern -> rows returned


class EnvelopeUnwrapper:
    def unwrap(
        self,
        answer: Mapping[str, Any],
        *,
        schema: SchemaDefinition,
        item: WorkItem,
        model: str | None,
        only: tuple[FieldPath, ...] | None = None,
    ) -> Unwrapped:
        walk = _Walk(SchemaTree(schema), Selection(SchemaTree(schema), only), item, model)
        for spec in walk.tree.roots():
            if walk.selection.keeps(spec.path.parts()):
                walk.node(spec, answer.get(spec.path.leaf_name), ())
        return Unwrapped(tuple(walk.candidates), tuple(walk.nulls), dict(walk.row_counts))


class _Walk:
    def __init__(
        self, tree: SchemaTree, selection: Selection, item: WorkItem, model: str | None
    ) -> None:
        self.tree, self.selection, self.item, self.model = tree, selection, item, model
        self.candidates: list[CandidateValue] = []
        self.nulls: list[FieldPath] = []
        self.row_counts: dict[str, int] = {}
        self.role = (
            CandidateRole.ALTERNATIVE if item.kind is WorkKind.CROSSCHECK else CandidateRole.PRIMARY
        )

    def node(self, spec: FieldSpec, value: Any, indices: tuple[int, ...]) -> None:
        if spec.json_type == "object":
            data = value if isinstance(value, Mapping) else {}
            for child in self.tree.children(spec.path.parts()):
                if self.selection.keeps(child.path.parts()):
                    self.node(child, data.get(child.path.leaf_name), indices)
        elif spec.json_type == "array":
            self.array(spec, value if isinstance(value, list) else [], indices)
        else:
            self.leaf(spec, value, indices)

    def array(self, spec: FieldSpec, items: list[Any], indices: tuple[int, ...]) -> None:
        offset = self.item.row_window[0] if self.item.row_window else 0
        item_parts: Parts = (*spec.path.parts(), "*")
        if self.tree.is_rows(spec):
            kept = 0
            for row in items:
                if isinstance(row, Mapping) and self.row(
                    item_parts, row, (*indices, offset + kept)
                ):
                    kept += 1
            self.row_counts[spec.path.raw] = kept
            return
        scalar = self.tree.item_spec(spec)
        for k, entry in enumerate(items):
            if scalar is not None:
                self.leaf(scalar, entry, (*indices, offset + k))
            else:
                for child in self.tree.children(item_parts):
                    data = entry if isinstance(entry, Mapping) else {}
                    self.node(child, data.get(child.path.leaf_name), (*indices, offset + k))

    def row(self, item_parts: Parts, row: Mapping[str, Any], indices: tuple[int, ...]) -> bool:
        columns = self._columns(item_parts, row)
        if all(raw is None for _, raw in columns):
            return False  # an empty row is noise, not data
        quote = _text(row.get("_quote"))
        page = _page(row.get("_page"))
        for spec, raw in columns:
            self.candidates.append(self._candidate(spec, raw, quote, page, indices))
        return True

    def _columns(self, parts: Parts, row: Mapping[str, Any]) -> list[tuple[FieldSpec, Any]]:
        out: list[tuple[FieldSpec, Any]] = []
        for child in self.tree.children(parts):
            raw = row.get(child.path.leaf_name)
            if child.json_type == "object":
                out.extend(
                    self._columns(child.path.parts(), raw if isinstance(raw, Mapping) else {})
                )
            else:
                out.append((child, _value(raw)))
        return out

    def leaf(self, spec: FieldSpec, value: Any, indices: tuple[int, ...]) -> None:
        data = value if isinstance(value, Mapping) else {"value": value}
        raw = _value(data.get("value"))
        if raw is None:
            self.nulls.append(_concrete(spec.path, indices))
            return
        candidate = self._candidate(
            spec, raw, _text(data.get("quote")), _page(data.get("page")), indices
        )
        self.candidates.append(candidate)

    def _candidate(
        self,
        spec: FieldSpec,
        raw: str | None,
        quote: str | None,
        page: int | None,
        indices: tuple[int, ...],
    ) -> CandidateValue:
        return CandidateValue(
            path=_concrete(spec.path, indices),
            raw=raw,
            work_item_id=self.item.item_id,
            round=self.item.round,
            tier=self.item.tier,
            quote=quote,
            page=page,
            model=self.model,
            role=self.role,
            source_kind=self.item.kind,
        )


def _concrete(pattern: FieldPath, indices: tuple[int, ...]) -> FieldPath:
    path = pattern
    for index in indices:
        path = path.with_index(index)
    return path


def _value(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "true" if raw else "false"
    text = str(raw).strip()
    return text or None


def _text(raw: Any) -> str | None:
    return raw.strip() or None if isinstance(raw, str) else None


def _page(raw: Any) -> int | None:
    """1-based page from the model -> 0-based; anything else is unknown."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int) and raw >= 1:
        return raw - 1
    if isinstance(raw, str) and raw.strip().isdigit() and int(raw) >= 1:
        return int(raw) - 1
    return None
