"""First-round planning: one EXTRACT work item per section chunk, at the cheap tier.

Pages read by local OCR go as text (cheaper, and their values can be grounded). Only pages
nobody could read, or pages the user forced, are sent as images.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bhoot.domain.document import Document, PageKind
from bhoot.domain.modes import Tier
from bhoot.domain.pack import DocPack, SectionSpec
from bhoot.domain.schema import FieldPath
from bhoot.domain.work import WorkItem, WorkKind
from bhoot.engine.planning.chunkers import CHUNKERS, ChunkerFactory, build_chunker
from bhoot.engine.planning.selectors import SelectorBuilder
from bhoot.engine.registry import StrategyRegistry

IMAGE_KINDS = frozenset({PageKind.SCANNED, PageKind.MIXED})


@dataclass(frozen=True, slots=True)
class Plan:
    items: tuple[WorkItem, ...]
    warnings: tuple[str, ...] = ()


class WorkPlanner:
    def __init__(
        self,
        selectors: SelectorBuilder | None = None,
        chunkers: StrategyRegistry[ChunkerFactory] = CHUNKERS,
    ) -> None:
        self.selectors = selectors or SelectorBuilder()
        self.chunkers = chunkers

    def plan(
        self,
        pack: DocPack,
        doc: Document,
        *,
        user_pages: Iterable[int] | None = None,  # 0-based
        forced_image_pages: Iterable[int] = (),  # 0-based
        skip: Iterable[tuple[str, int]] = (),  # (section, page) already covered by Tier 0
    ) -> Plan:
        allowed = set(user_pages) if user_pages is not None else None
        forced = set(forced_image_pages)
        covered = set(skip)
        items: list[WorkItem] = []
        warnings: list[str] = []
        for section in pack.sections:
            pages = [
                i
                for i in self.selectors.build(section.pages).select(doc)
                if (allowed is None or i in allowed)
                and doc.page(i).kind is not PageKind.BLANK
                and (section.name, i) not in covered
            ]
            if not pages:
                if not any(s == section.name for s, _ in covered):
                    warnings.append(f"section {section.name!r}: no pages selected")
                continue
            chunker = build_chunker(section.chunking, self.chunkers)
            for n, chunk in enumerate(chunker.split(pages, doc)):
                items.append(self._item(section, chunk, n, doc, forced))
        return Plan(tuple(items), tuple(warnings))

    @staticmethod
    def _item(
        section: SectionSpec, chunk: tuple[int, ...], n: int, doc: Document, forced: set[int]
    ) -> WorkItem:
        use_images = any(doc.page(i).kind in IMAGE_KINDS or i in forced for i in chunk)
        paths = () if "*" in section.paths else tuple(FieldPath(p) for p in section.paths)
        return WorkItem(
            item_id=WorkItem.make_id(0, WorkKind.EXTRACT, section.name, chunk, n),
            round=0,
            kind=WorkKind.EXTRACT,
            section=section.name,
            page_indices=chunk,
            tier=Tier.CHEAP,
            model_role="cheap",
            paths=paths,
            use_images=use_images,
        )
