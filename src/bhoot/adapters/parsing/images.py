"""Page rendering, image preparation for vision models, and crops for crop repair."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, ImageSequence

from bhoot.adapters.parsing.factory import sniff_mime
from bhoot.adapters.parsing.image_parser import image_dpi
from bhoot.adapters.parsing.pdfium_parser import open_pdf
from bhoot.domain.document import BBox
from bhoot.ports.parsing import DocumentInput


def _png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _limit(img: Image.Image, max_side_px: int) -> Image.Image:
    if max(img.size) <= max_side_px:
        return img
    scale = max_side_px / max(img.size)
    return img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.Resampling.LANCZOS,
    )


class DefaultPageRenderer:
    """Renders a page of a PDF or an image frame to PNG at the requested DPI."""

    def __init__(self, *, max_side_px: int = 6000, max_image_pixels: int = 89_478_485) -> None:
        self.max_side_px = max_side_px
        Image.MAX_IMAGE_PIXELS = max_image_pixels  # decompression-bomb guard

    def render(self, inp: DocumentInput, page_index: int, *, dpi: int) -> bytes:
        if sniff_mime(inp.data[:1024]) == "application/pdf":
            pdf = open_pdf(inp)
            try:
                page = pdf[page_index]
                width, height = page.get_size()
                scale = min(dpi / 72, self.max_side_px / max(width, height))
                img = page.render(scale=scale).to_pil()
                page.close()
            finally:
                pdf.close()
            return _png(img.convert("RGB"))
        with Image.open(BytesIO(inp.data)) as source:
            frame = next(f for i, f in enumerate(ImageSequence.Iterator(source)) if i == page_index)
            frame = ImageOps.exif_transpose(frame.copy()) or frame
            scale = dpi / image_dpi(frame)
            if abs(scale - 1) > 0.05:
                frame = frame.resize(
                    (max(1, round(frame.width * scale)), max(1, round(frame.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            return _png(_limit(frame.convert("RGB"), self.max_side_px))


class PillowPreprocessor:
    """Prepares page images for vision models: grayscale, gentle contrast, size cap."""

    def prepare(self, png: bytes, *, max_side_px: int, grayscale: bool) -> bytes:
        with Image.open(BytesIO(png)) as source:
            img = source.convert("L") if grayscale else source.convert("RGB")
            img = ImageOps.autocontrast(img, cutoff=1)
            return _png(_limit(img, max_side_px))


class PillowCropRenderer:
    """Cuts a region out of a page image, e.g. one statement row for crop repair."""

    def crop(self, png: bytes, bbox_px: BBox, *, pad_px: int, full_width: bool) -> bytes:
        with Image.open(BytesIO(png)) as source:
            w, h = source.size
            x0 = 0 if full_width else max(0, int(bbox_px.x0) - pad_px)
            x1 = w if full_width else min(w, int(bbox_px.x1) + pad_px)
            y0 = max(0, int(bbox_px.y0) - pad_px)
            y1 = min(h, int(bbox_px.y1) + pad_px)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"crop box {bbox_px} is outside the {w}x{h} image")
            return _png(source.crop((x0, y0, x1, y1)).convert("RGB"))
