"""Planning nodes: the first round's work, Tier-0 code extraction, and repair rounds."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from bhoot.domain.candidate import CandidateValue
from bhoot.domain.document import Document
from bhoot.domain.modes import Tier
from bhoot.domain.pack import CodeExtractionContext, ValueParsers
from bhoot.domain.work import WorkItem, WorkKind
from bhoot.engine.nodes.base import BaseNode, Update
from bhoot.engine.normalize.dates import parse_date
from bhoot.engine.normalize.numbers import parse_amount
from bhoot.engine.planning import RepairPlanner, SelectorBuilder, WorkPlanner
from bhoot.engine.tables import EngineTableProvider
from bhoot.ports.parsing import DocumentInput


class PlanNode(BaseNode):
    name = "plan"

    async def run(self, state: Mapping[str, Any]) -> Update:
        doc: Document = state["document"]
        options = self.ctx.job.options
        pack = self.ctx.job.pack
        user_pages = [p - 1 for p in options.pages] if options.pages else None
        plan = WorkPlanner().plan(
            pack,
            doc,
            user_pages=user_pages,
            forced_image_pages=[p - 1 for p in options.force_vision_pages],
        )
        selectors = SelectorBuilder()
        section_pages = {s.name: selectors.build(s.pages).select(doc) for s in pack.sections}
        return {
            "work_items": plan.items,
            "section_pages": section_pages,
            "round": 0,
            "warnings": plan.warnings,
        }


class CodeExtractNode(BaseNode):
    """Tier 0: the pack's code reads what it can prove (self-checked values only). Sections it
    fully covers on a page are dropped from the LLM plan; rows it could not read cleanly on OCR
    pages become crop re-reads."""

    name = "code_extract"

    async def run(self, state: Mapping[str, Any]) -> Update:
        pack = self.ctx.job.pack
        if pack.code_extractor is None:
            return {}
        doc: Document = state["document"]
        data = await self.services.blobs.get(self.ctx.job.input_ref)
        tables = EngineTableProvider(
            self.services.tables, DocumentInput(data, password=self.ctx.job.options.password)
        )
        values = ValueParsers(
            amount=parse_amount, date=lambda raw: parse_date(raw, dayfirst=pack.locale.dayfirst)
        )
        candidates: list[CandidateValue] = []
        covered: set[tuple[str, int]] = set()
        crops: list[WorkItem] = []
        notes: list[str] = []
        for section in pack.sections:
            ctx = CodeExtractionContext(
                doc, section, pack.locale, tables, values, self.ctx.pack_options
            )
            result = await asyncio.to_thread(pack.code_extractor.extract, ctx)
            candidates.extend(replace(c, tier=Tier.CODE, round=0) for c in result.candidates)
            covered |= set(result.covered_sections)
            notes.extend(result.notes)
            for n, req in enumerate(result.crop_requests):
                crops.append(
                    WorkItem(
                        item_id=WorkItem.make_id(0, WorkKind.CROP, section.name, (req.page,), n),
                        round=0,
                        kind=WorkKind.CROP,
                        section=section.name,
                        page_indices=(req.page,),
                        tier=Tier.REPAIR,
                        model_role="cheap",
                        use_images=True,
                        crops=(req,),
                        feedback=(req.reason,),
                    )
                )
        items = tuple(
            i
            for i in state.get("work_items", ())
            if not (
                i.kind is WorkKind.EXTRACT
                and all((i.section, p) in covered for p in i.page_indices)
            )
        )
        return {
            "work_items": (*items, *crops),
            "candidates": tuple(candidates),
            "warnings": tuple(notes),
        }


class RepairPlanNode(BaseNode):
    name = "repair_plan"

    async def run(self, state: Mapping[str, Any]) -> Update:
        items = RepairPlanner().plan(
            round_=int(state["round"]),
            decisions=state.get("decisions", ()),
            field_states=state.get("field_states", {}),
            doc=state["document"],
            pack=self.ctx.job.pack,
            doc_checks=state.get("doc_checks", ()),
            section_pages=state.get("section_pages"),
        )
        return {"work_items": items}
