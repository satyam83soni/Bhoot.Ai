"""Repair planning: turn this round's decisions into the next round's work items. Pure: no I/O.

* REPAIR -> the cheap model again, with feedback (failed checks, the value it gave, why it
  wasn't grounded). On OCR pages with a known box, a CROP item re-reads the region image.
* ESCALATE -> the strong model; a text page that already failed twice also gets its image
  (the text layer may be garbled).
* CROSS_CHECK -> a second model, with no feedback: it must answer independently.

Failing rows are asked for with their neighbours (``row_window``), because a running-balance
break implicates the rows around it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from bhoot.domain.checks import CheckResult
from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import CropRequest, Document, PageKind
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.modes import Tier
from bhoot.domain.pack import DocPack
from bhoot.domain.schema import FieldPath
from bhoot.domain.work import WorkItem, WorkKind
from bhoot.engine.merge.merger import array_root

IMAGE_KINDS = frozenset({PageKind.SCANNED, PageKind.MIXED})
_ROLE = {
    DecisionKind.REPAIR: (Tier.REPAIR, "cheap", WorkKind.REPAIR),
    DecisionKind.ESCALATE: (Tier.STRONG, "strong", WorkKind.REPAIR),
    DecisionKind.CROSS_CHECK: (Tier.CROSS, "cross", WorkKind.CROSSCHECK),
}


def section_for(pack: DocPack, path: FieldPath) -> str:
    """The first section whose paths cover the field (``*`` covers everything)."""
    target = path.pattern().parts()
    for section in pack.sections:
        for raw in section.paths:
            if raw == "*":
                return section.name
            parts = FieldPath(raw).pattern().parts()
            if target[: len(parts)] == parts:
                return section.name
    return pack.sections[0].name


def field_page(fs: FieldState) -> int | None:
    cand = fs.current
    if cand is None:
        return None
    if cand.evidence is not None:
        return cand.evidence.page
    return cand.page


@dataclass
class _Group:
    kind: WorkKind
    tier: Tier
    role: str
    section: str
    pages: set[int] = field(default_factory=set)
    paths: list[FieldPath] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    crops: list[CropRequest] = field(default_factory=list)
    rows: set[int] = field(default_factory=set)
    row_count: int = 0
    images: bool = False


class RepairPlanner:
    def plan(
        self,
        *,
        round_: int,
        decisions: Sequence[Decision],
        field_states: Mapping[str, FieldState],
        doc: Document,
        pack: DocPack,
        doc_checks: Sequence[CheckResult] = (),
        section_pages: Mapping[str, tuple[int, ...]] | None = None,
    ) -> tuple[WorkItem, ...]:
        groups: dict[tuple[object, ...], _Group] = {}
        row_counts = _row_counts(field_states)
        for decision in decisions:
            if decision.kind not in _ROLE:
                continue
            fs = field_states[decision.path.raw]
            tier, role, kind = _ROLE[decision.kind]
            section = section_for(pack, fs.path)
            page = field_page(fs)
            pages = (
                (page,)
                if page is not None
                else (section_pages or {}).get(section) or tuple(range(doc.page_count))
            )
            crop = self._crop(decision, fs, doc, page)
            if crop is not None:
                kind = WorkKind.CROP
            root = array_root(fs.path)
            key = (kind, tier, section, root.raw if root else None, None if root else pages)
            group = groups.setdefault(key, _Group(kind, tier, role, section))
            group.pages.update(pages)
            group.paths.append(fs.path)
            if root is not None:
                index = fs.path.parts()[len(root.parts())]
                assert isinstance(index, int)
                group.rows.add(index)
                group.row_count = row_counts.get(root.raw, index + 1)
            if crop is not None:
                group.crops.append(crop)
            if kind is not WorkKind.CROSSCHECK:
                group.feedback.extend(_feedback(fs, doc_checks))
            group.images |= kind is WorkKind.CROP or any(
                doc.page(p).kind in IMAGE_KINDS for p in pages
            )
            if decision.kind is DecisionKind.ESCALATE and len(fs.attempts) >= 2:
                group.images = True

        items: list[WorkItem] = []
        counters: dict[tuple[WorkKind, str], int] = defaultdict(int)
        for group in groups.values():
            window = None
            page_set = set(group.pages)
            if group.rows:
                start = max(0, min(group.rows) - 1)
                end = min(group.row_count, max(group.rows) + 2)
                window = (start, end)
                page_set |= _row_pages(field_states, group.paths, window)
            ordered = tuple(sorted(page_set))
            n = counters[(group.kind, group.section)]
            counters[(group.kind, group.section)] += 1
            items.append(
                WorkItem(
                    item_id=WorkItem.make_id(round_, group.kind, group.section, ordered, n),
                    round=round_,
                    kind=group.kind,
                    section=group.section,
                    page_indices=ordered,
                    tier=group.tier,
                    model_role=group.role,
                    paths=tuple(dict.fromkeys(group.paths)),
                    use_images=group.images,
                    feedback=tuple(dict.fromkeys(group.feedback)),
                    crops=tuple(group.crops),
                    row_window=window,
                )
            )
        return tuple(items)

    @staticmethod
    def _crop(
        decision: Decision, fs: FieldState, doc: Document, page: int | None
    ) -> CropRequest | None:
        if decision.kind is not DecisionKind.REPAIR or page is None:
            return None
        cand = fs.current
        evidence = cand.evidence if cand else None
        if (
            evidence is None
            or evidence.bbox is None
            or doc.page(page).kind is not PageKind.OCR_TEXT
        ):
            return None
        return CropRequest(page, evidence.bbox, (fs.path.raw,), decision.reason)


def _feedback(fs: FieldState, doc_checks: Iterable[CheckResult]) -> list[str]:
    lines = [
        c.message
        for c in (*fs.checks, *doc_checks)
        if c.blocks and c.message and c.implicates(fs.path)
    ]
    if fs.status is FieldStatus.PENDING:
        need = " (it is required)" if fs.spec.required else ""
        lines.append(f"No value was returned for {fs.path}{need}: look for it again.")
    elif fs.current is not None:
        lines.append(
            f"You returned {fs.path} = {fs.current.raw!r}: read it again exactly as printed."
        )
        if fs.status is FieldStatus.UNGROUNDED and fs.history and fs.history[-1][2]:
            lines.append(f"{fs.path}: {fs.history[-1][2]}.")
    return lines


def _row_counts(field_states: Mapping[str, FieldState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fs in field_states.values():
        root = array_root(fs.path)
        if root is not None:
            index = fs.path.parts()[len(root.parts())]
            assert isinstance(index, int)
            counts[root.raw] = max(counts.get(root.raw, 0), index + 1)
    return counts


def _row_pages(
    field_states: Mapping[str, FieldState], paths: Sequence[FieldPath], window: tuple[int, int]
) -> set[int]:
    roots = {array_root(p) for p in paths}
    pages: set[int] = set()
    for fs in field_states.values():
        root = array_root(fs.path)
        if root in roots and root is not None:
            index = fs.path.parts()[len(root.parts())]
            if isinstance(index, int) and window[0] <= index < window[1]:
                page = field_page(fs)
                if page is not None:
                    pages.add(page)
    return pages
