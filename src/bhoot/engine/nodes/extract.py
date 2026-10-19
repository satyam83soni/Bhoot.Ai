"""``llm_extract``: the only node that calls a model (one branch per work item).

The branch gets the work item and light copies of its pages (no character boxes), so pending
branches stay small in checkpoints. It reserves the estimated cost before calling and settles
the real cost after; model errors are recorded on the item (its fields then go through the
policy), except a provider authentication error, which stops the job.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from bhoot.domain.cost import CostEntry, CostReport
from bhoot.domain.document import BBox, Page
from bhoot.domain.errors import LLMError, ModelConfigError, ProviderAuthError, ResponseParseError
from bhoot.domain.schema import Granularity
from bhoot.domain.work import WorkItem, WorkKind, WorkOutcome, WorkStatus
from bhoot.engine.cost.pricing import CallShape, estimate_usd
from bhoot.engine.events import BudgetExhausted, LLMCalled
from bhoot.engine.nodes.base import BaseNode, Update
from bhoot.engine.prompting import EXAMPLE_SELECTORS, MessageBuilder, PromptInputs
from bhoot.engine.schema_compiler import (
    EnvelopeUnwrapper,
    EvidenceSchemaBuilder,
    SchemaTree,
    Selection,
    dialect_for,
    parse_answer,
)
from bhoot.ports.llm import LLMMessage, LLMRequest
from bhoot.ports.parsing import DocumentInput

CROP_PAD_PX = 8
ROWS_PER_IMAGE_PAGE = 40


def light_page(page: Page) -> Page:
    """A page without character boxes, words or tables: all a model call needs."""
    return replace(page, chars=(), words=(), tables=())


@dataclass(frozen=True)
class PageView:
    pages: Mapping[int, Page]

    def page(self, index: int) -> Page:
        return self.pages[index]


def png_size(png: bytes) -> tuple[int, int]:
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


class LLMExtractNode(BaseNode):
    name = "llm_extract"

    def timing_key(self, payload: Mapping[str, Any]) -> str:
        return f"llm_extract:{payload['item'].item_id}"

    async def run(self, payload: Mapping[str, Any]) -> Update:
        item: WorkItem = payload["item"]
        view = PageView({p.index: p for p in payload["pages"]})
        services, job = self.services, self.ctx.job
        pack, schema = job.pack, job.pack.schema
        if services.models is None:
            raise ModelConfigError("no models configured: add a models: section to bhoot.yaml")
        llm = services.llm
        spec = services.models.resolve(
            item.model_role,  # type: ignore[arg-type]
            needs_vision=item.use_images or item.kind is WorkKind.CROP,
            has_vision=lambda model_id: llm.capabilities(model_id).vision,
        )
        model = spec.model_id
        dialect, warning = dialect_for(llm.capabilities(model))
        only = item.paths or None
        envelope = EvidenceSchemaBuilder().build(schema, only=only)
        if item.use_images:
            view = await self._with_images(item, view)
        crop_refs = await self._render_crops(item, view)
        inputs = PromptInputs(
            schema, pack.prompt, pack.examples, pack.locale, self.ctx.pack_options
        )
        selector = EXAMPLE_SELECTORS.get(services.settings.engine.example_selector)
        messages = MessageBuilder(selector).build(
            inputs, item, view, envelope=envelope, dialect=dialect, crop_refs=crop_refs
        )
        request = LLMRequest(
            model=model,
            messages=messages,
            response_format=dialect.adapt(envelope, f"{pack.name}_{item.section}"),
            metadata={
                "job": job.job_id,
                "item": item.item_id,
                "section": item.section,
                "kind": item.kind.value,
                "tier": str(int(item.tier)),
                "round": str(item.round),
                "paths": ",".join(p.raw for p in item.paths),
            },
        )
        warnings = (warning,) if warning else ()
        estimate = estimate_usd(
            services.prices.price(model), self._shape(item, view, messages, schema)
        )
        reservation = await services.budget.reserve(job.job_id, job.budget, estimate)
        if reservation is None:
            remaining = services.budget.remaining(job.job_id, job.budget)
            services.events.publish(BudgetExhausted(job.job_id, item.item_id, estimate, remaining))
            outcome = WorkOutcome(
                item.item_id, WorkStatus.SKIPPED_BUDGET, model, "budget exhausted"
            )
            return {"work_outcomes": {item.item_id: outcome}, "warnings": warnings}
        try:
            response = await llm.complete(request)
        except ProviderAuthError:
            await services.budget.release(reservation)
            raise
        except LLMError as exc:
            await services.budget.release(reservation)
            outcome = WorkOutcome(
                item.item_id, WorkStatus.ERROR, model, f"{type(exc).__name__}: {exc}"[:300]
            )
            return {"work_outcomes": {item.item_id: outcome}, "warnings": warnings}
        await services.budget.settle(reservation, response.cost_usd)
        entry = CostEntry(
            "llm_extract", model, item.tier, item.item_id, response.usage,
            response.cost_usd, response.latency_ms, response.cache_hit,
        )  # fmt: skip
        services.events.publish(
            LLMCalled(job.job_id, item.item_id, model, item.tier, response.usage,
                      response.cost_usd, response.latency_ms, response.cache_hit)
        )  # fmt: skip
        cost = CostReport.of(entry)
        try:
            unwrapped = EnvelopeUnwrapper().unwrap(
                parse_answer(response), schema=schema, item=item, model=model, only=only
            )
        except ResponseParseError as exc:
            outcome = WorkOutcome(item.item_id, WorkStatus.ERROR, model, str(exc)[:300])
            return {"work_outcomes": {item.item_id: outcome}, "cost": cost, "warnings": warnings}
        return {
            "candidates": unwrapped.candidates,
            "nulls": unwrapped.nulls,
            "work_outcomes": {item.item_id: WorkOutcome(item.item_id, WorkStatus.DONE, model)},
            "cost": cost,
            "warnings": warnings,
        }

    # ---------------------------------------------------------------- images
    async def _page_png(self, page: Page) -> tuple[bytes, Page]:
        """The page's image, rendering (and storing) it first if no step has yet."""
        blobs = self.services.blobs
        if page.image_ref is not None:
            return await blobs.get(page.image_ref), page
        job, vision = self.ctx.job, self.services.settings.vision
        inp = DocumentInput(await blobs.get(job.input_ref), password=job.options.password)
        png = await asyncio.to_thread(
            self.services.renderer.render, inp, page.index, dpi=vision.dpi
        )
        png = await asyncio.to_thread(
            self.services.preprocessor.prepare,
            png,
            max_side_px=vision.max_side_px,
            grayscale=vision.grayscale,
        )
        ref = await blobs.put(png, namespace="pages", suffix=".png")
        return png, replace(page, image_ref=ref)

    async def _with_images(self, item: WorkItem, view: PageView) -> PageView:
        pages = dict(view.pages)
        for index in item.page_indices:
            if pages[index].image_ref is None:
                _, pages[index] = await self._page_png(pages[index])
        return PageView(pages)

    async def _render_crops(self, item: WorkItem, view: PageView) -> tuple[str, ...]:
        refs: list[str] = []
        for request in item.crops:
            png, _ = await self._page_png(view.page(request.page))
            width, _height = png_size(png)
            scale = width / view.page(request.page).width
            box = request.bbox
            bbox_px = BBox(box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale)
            crop = await asyncio.to_thread(
                self.services.crops.crop, png, bbox_px, pad_px=CROP_PAD_PX, full_width=True
            )
            refs.append(await self.services.blobs.put(crop, namespace="crops", suffix=".png"))
        return tuple(refs)

    # ---------------------------------------------------------------- estimates
    @staticmethod
    def _shape(
        item: WorkItem, view: PageView, messages: tuple[LLMMessage, ...], schema: Any
    ) -> CallShape:
        text_chars = sum(len(p.text or "") for m in messages for p in m.parts if p.type == "text")
        images = sum(1 for m in messages for p in m.parts if p.type == "image")
        tree = SchemaTree(schema)
        selection = Selection(tree, item.paths or None)
        leaves = row_fields = 0
        has_rows = False
        for spec in schema.fields.values():
            if not selection.keeps(spec.path.parts()) or spec.json_type in ("object", "array"):
                continue
            if spec.granularity is Granularity.ROW:
                has_rows = True
                row_fields += 1
            else:
                leaves += 1
        rows = 0
        if has_rows:
            if item.row_window is not None:
                rows = item.row_window[1] - item.row_window[0]
            else:
                for index in item.page_indices:
                    text = view.page(index).text
                    rows += text.count("\n") + 1 if text.strip() else ROWS_PER_IMAGE_PAGE
        return CallShape(
            text_chars=text_chars,
            images=images,
            leaves=leaves,
            expected_rows=rows,
            row_fields=row_fields,
        )
