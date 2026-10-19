"""Reading the document: parse it, then decide how each page can be read and render the pages
a model will have to look at."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from bhoot.domain.document import Document, Page, PageKind
from bhoot.engine.nodes.base import BaseNode, Update
from bhoot.ports.parsing import DocumentInput

MIN_TEXT_CHARS = 20  # fewer letters/digits than this: the text layer is not usable
MAX_GARBLED_SHARE = 0.10  # more replacement/control characters than this: broken text layer
IMAGE_KINDS = frozenset({PageKind.SCANNED, PageKind.MIXED})


def classify_page(page: Page) -> PageKind:
    text = page.text
    useful = sum(ch.isalnum() for ch in text)
    if useful < MIN_TEXT_CHARS:
        return PageKind.SCANNED
    garbled = sum(ch == "�" or (not ch.isprintable() and not ch.isspace()) for ch in text)
    if garbled / max(1, len(text)) > MAX_GARBLED_SHARE:
        return PageKind.MIXED
    return PageKind.TEXT


class LoadNode(BaseNode):
    name = "load"

    async def run(self, state: Mapping[str, Any]) -> Update:
        job = self.ctx.job
        data = await self.services.blobs.get(job.input_ref)
        inp = DocumentInput(data, filename=None, password=job.options.password)
        parser = self.services.parsers.for_input(inp)
        limit = min(
            job.options.max_pages or self.services.settings.limits.max_pages,
            self.services.settings.limits.max_pages,
        )
        doc = await asyncio.to_thread(parser.parse, inp, max_pages=limit)
        return {"document": doc}


class ClassifyNode(BaseNode):
    """Page kinds, plus page images (stored by reference) for every page a model will see as
    an image: scanned or broken pages, and pages the user forced."""

    name = "classify"

    async def run(self, state: Mapping[str, Any]) -> Update:
        doc: Document = state["document"]
        forced = {p - 1 for p in self.ctx.job.options.force_vision_pages}
        vision = self.services.settings.vision
        data: bytes | None = None
        pages: list[Page] = []
        for page in doc.pages:
            kind = page.kind if page.kind is not PageKind.UNCLASSIFIED else classify_page(page)
            page = replace(page, kind=kind)
            if (kind in IMAGE_KINDS or page.index in forced) and page.image_ref is None:
                if data is None:
                    data = await self.services.blobs.get(self.ctx.job.input_ref)
                inp = DocumentInput(data, password=self.ctx.job.options.password)
                png = await asyncio.to_thread(
                    self.services.renderer.render, inp, page.index, dpi=vision.dpi
                )
                png = await asyncio.to_thread(
                    self.services.preprocessor.prepare,
                    png,
                    max_side_px=vision.max_side_px,
                    grayscale=vision.grayscale,
                )
                ref = await self.services.blobs.put(png, namespace="pages", suffix=".png")
                page = replace(page, image_ref=ref)
            pages.append(page)
        return {"document": doc.with_pages(tuple(pages))}
