"""Apply one round of candidates to the field states.

* Only candidates from this round are used.
* ``ALTERNATIVE`` answers (cross-checks) are stored next to the value, never replacing it.
* Scalars: one candidate per path, picked by the scalar strategy.
* Rows: answers without a row window are combined in page order (first extraction); an
  answer for ``row_window=(a, b)`` replaces exactly rows a..b-1, shifting later rows when the
  count changed (with a warning; the row rules then re-check the chain).
* A field that already has a final decision is never overwritten.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from bhoot.domain.candidate import CandidateRole, CandidateValue
from bhoot.domain.field_state import FieldState
from bhoot.domain.schema import FieldPath, Granularity, SchemaDefinition
from bhoot.domain.work import WorkItem
from bhoot.engine.merge.strategies import (
    LIST_STRATEGIES,
    SCALAR_STRATEGIES,
    ItemOrder,
    ListStrategy,
    Row,
    ScalarStrategy,
    relative,
)
from bhoot.engine.registry import StrategyRegistry


@dataclass(frozen=True)
class MergeResult:
    field_states: dict[str, FieldState]
    returned_null: frozenset[str]  # leaf paths a model answered with null this round
    warnings: tuple[str, ...]


def initial_states(schema: SchemaDefinition) -> dict[str, FieldState]:
    """A PENDING state for every scalar outside a list (rows appear as they are read)."""
    return {
        spec.path.raw: FieldState(spec.path, spec)
        for spec in schema.leaf_specs()
        if not spec.path.is_pattern()
    }


def array_root(path: FieldPath) -> FieldPath | None:
    parts = path.parts()
    for i, part in enumerate(parts):
        if isinstance(part, int):
            return FieldPath.from_parts(parts[:i])
    return None


class Merger:
    def __init__(
        self,
        scalars: StrategyRegistry[ScalarStrategy] = SCALAR_STRATEGIES,
        lists: StrategyRegistry[ListStrategy] = LIST_STRATEGIES,
        scalar_strategy: str = "scalar_prefer_verified",
    ) -> None:
        self.scalar = scalars.get(scalar_strategy)
        self.lists = lists

    def merge(
        self,
        *,
        schema: SchemaDefinition,
        field_states: Mapping[str, FieldState],
        candidates: Iterable[CandidateValue],
        items: Sequence[WorkItem],
        nulls: Iterable[FieldPath] = (),
        round_: int,
    ) -> MergeResult:
        states = {**initial_states(schema), **field_states}
        order = {
            item.item_id: ItemOrder(item.page_indices[0] if item.page_indices else 0, n)
            for n, item in enumerate(items)
        }
        windows = {item.item_id: item.row_window for item in items if item.row_window is not None}
        warnings: list[str] = []

        scalars: dict[str, list[CandidateValue]] = defaultdict(list)
        arrays: dict[str, dict[str, dict[int, Row]]] = defaultdict(lambda: defaultdict(dict))
        for cand in candidates:
            if cand.round != round_:
                continue
            if cand.role is CandidateRole.ALTERNATIVE:
                current = states.get(cand.path.raw)
                if current is not None and not current.is_terminal:
                    states[cand.path.raw] = current.with_alternative(cand)
                continue
            root = array_root(cand.path)
            if root is None:
                scalars[cand.path.raw].append(cand)
            else:
                index = cand.path.parts()[len(root.parts())]
                assert isinstance(index, int)
                arrays[root.raw][cand.work_item_id].setdefault(index, {})[
                    relative(cand.path, root)
                ] = cand

        for path, found in scalars.items():
            current = states.get(path)
            if current is not None and current.is_terminal:
                continue
            best = self.scalar.pick(found, order)
            base = current or FieldState(best.path, schema.spec_for(best.path))
            states[path] = base.with_candidate(best)

        for root_raw, by_item in arrays.items():
            root = FieldPath(root_raw)
            whole = [i for i in by_item if i not in windows]
            windowed = sorted((i for i in by_item if i in windows), key=lambda i: windows[i])
            rows = _current_rows(states, root)
            if whole:
                strategy = self.lists.get(_list_strategy(schema, root))
                pairs = [(order[i], [by_item[i][k] for k in sorted(by_item[i])]) for i in whole]
                combined, notes = strategy.combine(pairs)
                warnings.extend(f"{root_raw}: {n}" for n in notes)
                rows = [{col: _apply(old.get(col) if old else None, c, schema) for col, c in new.items()}
                        for old, new in _zip_longest(rows, combined)]  # fmt: skip
            shift = 0
            for item_id in windowed:
                start, end = windows[item_id]
                start, end = start + shift, end + shift
                new_rows = [by_item[item_id][k] for k in sorted(by_item[item_id])]
                old_rows = rows[start:end]
                replaced = [
                    {
                        col: _apply(old.get(col) if old else None, c, schema)
                        for col, c in new.items()
                    }
                    for old, new in _zip_longest(old_rows, new_rows)
                ]
                if len(new_rows) != end - start:
                    warnings.append(
                        f"{root_raw}: rows {start + 1}-{end} were re-read as {len(new_rows)} rows; "
                        "later rows were renumbered"
                    )
                rows[start:end] = replaced
                shift += len(new_rows) - (end - start)
            states = _rekey(states, root, rows)

        nulls_now = frozenset(p.raw for p in nulls)
        return MergeResult(states, nulls_now, tuple(warnings))


def _list_strategy(schema: SchemaDefinition, root: FieldPath) -> str:
    spec = schema.fields.get(root.pattern().raw)
    return (
        "rows_by_page"
        if spec is not None and spec.granularity is Granularity.ROW
        else "list_concat"
    )


def _zip_longest(
    old: list[dict[str, FieldState]], new: list[Row]
) -> list[tuple[dict[str, FieldState] | None, Row]]:
    return [(old[i] if i < len(old) else None, row) for i, row in enumerate(new)]


def _apply(old: FieldState | None, cand: CandidateValue, schema: SchemaDefinition) -> FieldState:
    if old is not None and old.is_terminal and old.value == cand.value and old.current is not None:
        return old
    base = old if old is not None else FieldState(cand.path, schema.spec_for(cand.path))
    return base.with_candidate(cand)


def _current_rows(states: Mapping[str, FieldState], root: FieldPath) -> list[dict[str, FieldState]]:
    rows: dict[int, dict[str, FieldState]] = defaultdict(dict)
    for fs in states.values():
        if array_root(fs.path) == root:
            index = fs.path.parts()[len(root.parts())]
            assert isinstance(index, int)
            rows[index][relative(fs.path, root)] = fs
    return [rows.get(i, {}) for i in range(max(rows) + 1)] if rows else []


def _rekey(
    states: dict[str, FieldState], root: FieldPath, rows: list[dict[str, FieldState]]
) -> dict[str, FieldState]:
    out = {p: fs for p, fs in states.items() if array_root(fs.path) != root}
    for index, row in enumerate(rows):
        for col, fs in row.items():
            parts = (*root.parts(), index, *(FieldPath(col).parts() if col else ()))
            path = FieldPath.from_parts(parts)
            if fs.path != path:
                current = replace(fs.current, path=path) if fs.current is not None else None
                fs = replace(fs, path=path, current=current)
            out[path.raw] = fs
    return out
