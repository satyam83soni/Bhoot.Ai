"""Reading the document: parse it, decide how each page can be read, read scanned pages with
local OCR, and render images only for the pages a vision model will have to look at."""

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
    """How each page can be read: a usable text layer, a broken one, or none (scanned)."""

    name = "classify"

    async def run(self, state: Mapping[str, Any]) -> Update:
        doc: Document = state["document"]
        pages = tuple(
            replace(p, kind=classify_page(p)) if p.kind is PageKind.UNCLASSIFIED else p
            for p in doc.pages
        )
        return {"document": doc.with_pages(pages)}


class OcrNode(BaseNode):
    """Read scanned and broken pages with local OCR. A page whose OCR confidence reaches
    ``ocr.min_page_confidence`` becomes OCR_TEXT: it is then sent to the model as text (much
    cheaper than an image) and its values can be grounded on the page. Other pages stay
    SCANNED and go to the vision model."""

    name = "ocr"

    async def run(self, state: Mapping[str, Any]) -> Update:
        doc: Document = state["document"]
        targets = [p for p in doc.pages if p.kind in IMAGE_KINDS]
        if not targets:
            return {}
        services, settings = self.services, self.services.settings.ocr
        engine = services.ocr
        if not settings.enabled or engine is None or not engine.available():
            reason = "disabled in bhoot.yaml" if not settings.enabled else "install bhoot[ocr]"
            return {
                "warnings": (
                    f"{len(targets)} page(s) have no text layer and will be read by a vision model "
                    f"(local OCR {reason} for cheaper, grounded reading)",
                )
            }
        assert services.scan_preprocessor is not None
        assert services.ocr_assembler is not None
        data = await services.blobs.get(self.ctx.job.input_ref)
        inp = DocumentInput(data, password=self.ctx.job.options.password)
        limit = asyncio.Semaphore(settings.max_parallel_pages or 2)

        async def read(page: Page) -> Page:
            async with limit:
                return await asyncio.to_thread(self._read_page, inp, page)

        read_pages = {p.index: p for p in await asyncio.gather(*(read(p) for p in targets))}
        pages = tuple(read_pages.get(p.index, p) for p in doc.pages)
        low = [p.number for p in read_pages.values() if p.kind is not PageKind.OCR_TEXT]
        warnings = (
            (f"OCR confidence too low on page(s) {low}: read by a vision model instead",)
            if low
            else ()
        )
        return {"document": doc.with_pages(pages), "warnings": warnings}

    def _read_page(self, inp: DocumentInput, page: Page) -> Page:
        services, settings = self.services, self.services.settings.ocr
        assert services.ocr is not None
        assert services.scan_preprocessor is not None
        assert services.ocr_assembler is not None
        png = services.renderer.render(inp, page.index, dpi=settings.dpi)
        prepared = services.scan_preprocessor.prepare(png, mode=settings.preprocess)
        result = services.ocr.recognize(prepared.png, lang=settings.lang)
        text = services.ocr_assembler.assemble(
            result, prepared, dpi=settings.dpi, page_width_pt=page.width, page_height_pt=page.height
        )
        readable = bool(text.words) and text.mean_confidence >= settings.min_page_confidence
        if not readable:
            return replace(page, ocr_confidence=text.mean_confidence)
        return replace(
            page,
            kind=PageKind.OCR_TEXT,
            text=text.text,
            chars=text.chars,
            words=text.words,
            text_source="ocr",
            ocr_confidence=text.mean_confidence,
        )


class PageImagesNode(BaseNode):
    """Render images (stored by reference) for every page a model will see as an image:
    pages nobody could read, and pages the user forced."""

    name = "page_images"

    async def run(self, state: Mapping[str, Any]) -> Update:
        doc: Document = state["document"]
        forced = {p - 1 for p in self.ctx.job.options.force_vision_pages}
        need = [
            p
            for p in doc.pages
            if (p.kind in IMAGE_KINDS or p.index in forced) and p.image_ref is None
        ]
        if not need:
            return {}
        vision = self.services.settings.vision
        data = await self.services.blobs.get(self.ctx.job.input_ref)
        inp = DocumentInput(data, password=self.ctx.job.options.password)
        rendered: dict[int, Page] = {}
        for page in need:
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
            rendered[page.index] = replace(page, image_ref=ref)
        return {"document": doc.with_pages(tuple(rendered.get(p.index, p) for p in doc.pages))}
