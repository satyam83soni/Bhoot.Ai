"""Turn digital statements into realistic scans: image-only PDFs for OCR tests.

Each profile is a small pipeline of image filters applied page by page: stamp → rotation →
perspective warp → uneven lighting → blur → sensor noise → binarisation → JPEG compression.
Everything is seeded, so the same document, profile and seed always give the same bytes.
The ground truth does not change: a scan prints exactly the same ledger.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FIXED_PDF_DATE = time.strptime("2026-01-01", "%Y-%m-%d")  # fixed PDF dates: reproducible output


@dataclass(frozen=True)
class ScanProfile:
    name: str
    dpi: int
    max_rotation_deg: float
    blur_radius: float
    noise_sigma: float
    jpeg_quality: int
    perspective: float = 0.0  # max corner shift, fraction of page size (phone photos)
    shadow: float = 0.0  # strength of uneven lighting, 0..1
    binarize: bool = False  # fax-like black and white
    speckle: float = 0.0  # fraction of pixels flipped to black/white (fax dirt)
    stamp: bool = False


PROFILES: dict[str, ScanProfile] = {
    p.name: p
    for p in (
        ScanProfile(
            "clean", dpi=200, max_rotation_deg=0.4, blur_radius=0.3, noise_sigma=3, jpeg_quality=85
        ),
        ScanProfile(
            "phone_photo",
            dpi=170,
            max_rotation_deg=2.0,
            blur_radius=0.9,
            noise_sigma=6,
            jpeg_quality=70,
            perspective=0.025,
            shadow=0.35,
        ),
        ScanProfile(
            "fax",
            dpi=150,
            max_rotation_deg=1.0,
            blur_radius=0.5,
            noise_sigma=8,
            jpeg_quality=75,
            binarize=True,
            speckle=0.002,
        ),
        ScanProfile(
            "stamped",
            dpi=200,
            max_rotation_deg=0.8,
            blur_radius=0.4,
            noise_sigma=4,
            jpeg_quality=80,
            stamp=True,
        ),
        ScanProfile(
            "low_dpi",
            dpi=110,
            max_rotation_deg=0.6,
            blur_radius=0.6,
            noise_sigma=4,
            jpeg_quality=60,
        ),
    )
}

STAMP_WORDS = ("RECEIVED", "VERIFIED", "CERTIFIED TRUE COPY", "BRANCH COPY")


class ScanSimulator:
    def __init__(self, profile: ScanProfile, seed: int) -> None:
        self.profile = profile
        self.seed = seed

    def simulate(self, pdf: bytes) -> bytes:
        doc = pdfium.PdfDocument(pdf)
        try:
            scale = self.profile.dpi / 72
            pages = [
                self.page(doc[i].render(scale=scale, grayscale=True).to_pil(), page_index=i)
                for i in range(len(doc))
            ]
        finally:
            doc.close()

        out = BytesIO()
        pages[0].save(
            out,
            "PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=float(self.profile.dpi),
            title="Scanned statement (synthetic)",
            producer="bhoot synthetic scan simulator",
            creationDate=FIXED_PDF_DATE,
            modDate=FIXED_PDF_DATE,
        )
        return out.getvalue()

    # ---------------------------------------------------------------- one page
    def page(self, image: Image.Image, *, page_index: int) -> Image.Image:
        p = self.profile
        rng = random.Random(self.seed * 1_000 + page_index)
        nrng = np.random.default_rng(self.seed * 1_000 + page_index)

        img = image.convert("L")
        if p.stamp and page_index == 0:
            img = self._stamp(img, rng)
        if p.max_rotation_deg:
            angle = rng.uniform(-p.max_rotation_deg, p.max_rotation_deg)
            img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        if p.perspective:
            img = self._perspective(img, rng)
        if p.shadow:
            img = self._shadow(img, rng)
        if p.blur_radius:
            img = img.filter(ImageFilter.GaussianBlur(p.blur_radius))
        if p.noise_sigma:
            arr = np.asarray(img, dtype=np.float32)
            arr += nrng.normal(0.0, p.noise_sigma, arr.shape).astype(np.float32)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
        if p.binarize:
            img = img.point(lambda v: 255 if v > 160 else 0)
            if p.speckle:
                arr = np.array(img)
                mask = nrng.random(arr.shape) < p.speckle
                arr[mask] = 255 - arr[mask]
                img = Image.fromarray(arr, mode="L")
        return self._jpeg(img, p.jpeg_quality)

    # ---------------------------------------------------------------- filters
    @staticmethod
    def _jpeg(img: Image.Image, quality: int) -> Image.Image:
        buf = BytesIO()
        img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("L")

    def _perspective(self, img: Image.Image, rng: random.Random) -> Image.Image:
        w, h = img.size
        shift = self.profile.perspective * min(w, h)

        def jitter(x: float, y: float) -> tuple[float, float]:
            return x + rng.uniform(-shift, shift), y + rng.uniform(-shift, shift)

        out_corners = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
        in_corners = [jitter(x, y) for x, y in out_corners]
        coeffs = _perspective_coefficients(out_corners, in_corners)
        return img.transform(
            (w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC, fillcolor=235
        )

    def _shadow(self, img: Image.Image, rng: random.Random) -> Image.Image:
        w, h = img.size
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / np.hypot(w, h)
        light = 1.0 - self.profile.shadow * np.clip(dist * 1.6, 0.0, 1.0)
        arr = np.asarray(img, dtype=np.float32) * light
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

    @staticmethod
    def _stamp(img: Image.Image, rng: random.Random) -> Image.Image:
        w, h = img.size
        size = int(min(w, h) * 0.22)
        layer = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(layer)
        draw.ellipse((4, 4, size - 4, size - 4), outline=255, width=max(3, size // 40))
        draw.ellipse(
            (size * 0.12, size * 0.12, size * 0.88, size * 0.88),
            outline=255,
            width=max(2, size // 60),
        )
        font = ImageFont.load_default(size=max(10, size // 9))
        word = rng.choice(STAMP_WORDS)
        draw.text((size / 2, size / 2), word, fill=255, font=font, anchor="mm")
        layer = layer.rotate(rng.uniform(-25, 25), resample=Image.Resampling.BICUBIC, fillcolor=0)

        # place over the table area on the first page, partly covering text
        x = int(rng.uniform(0.45, 0.75) * w)
        y = int(rng.uniform(0.30, 0.60) * h)
        ink = Image.new("L", layer.size, 110)  # mid-grey ink (red stamp after grayscale)
        alpha = layer.point(lambda v: int(v * 0.7))
        out = img.copy()
        out.paste(ink, (x - size // 2, y - size // 2), alpha)
        return out


def _perspective_coefficients(
    out_corners: list[tuple[float, float]], in_corners: list[tuple[float, float]]
) -> tuple[float, ...]:
    """Coefficients for PIL's PERSPECTIVE transform (maps output pixels to input pixels)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(out_corners, in_corners, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    solution = np.linalg.solve(
        np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64)
    )
    return tuple(float(c) for c in solution)
