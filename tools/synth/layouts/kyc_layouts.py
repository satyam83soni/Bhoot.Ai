"""KYC specimen rendering.

Each document is laid out as one or two *panels* (front / back or data page / address page),
described once as text lines and drawn by two renderers:

* ``image``: a photocopy-like A4 scan (Pillow, no text layer) → OCR path
* ``digital``: an e-document PDF with a real text layer (reportlab) → text path

No emblems or official headers are drawn, and a "SPECIMEN" watermark is always present.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from tools.synth.kyc import KycDocument
from tools.synth.layouts.base import FONT, FONT_BOLD, MARGIN, SYNTHETIC_NOTICE, wrap_text
from tools.synth.scan_sim import FIXED_PDF_DATE

Renderer = Literal["image", "digital"]
WATERMARK = "SPECIMEN - NOT A VALID DOCUMENT"

TITLES = {
    "pan": "PERMANENT ACCOUNT NUMBER CARD",
    "aadhaar": "AADHAAR",
    "passport": "PASSPORT",
    "driving_licence": "DRIVING LICENCE",
    "voter_id": "ELECTOR PHOTO IDENTITY CARD",
}
TINTS = {
    "pan": (226, 238, 250),
    "aadhaar": (252, 244, 224),
    "passport": (230, 242, 232),
    "driving_licence": (240, 232, 248),
    "voter_id": (238, 238, 238),
}


@dataclass(frozen=True)
class Panel:
    title: str
    lines: tuple[tuple[str, str], ...]  # (label, value); empty label = free text line
    photo: bool = True


def _d(value: date | None) -> str:
    return value.strftime("%d/%m/%Y") if value else ""


def panels(doc: KycDocument) -> tuple[Panel, ...]:
    t = TITLES[doc.document_type]
    g = {"male": "Male", "female": "Female"}.get(doc.gender or "", "")
    if doc.document_type == "pan":
        return (Panel(t, (
            ("Name", doc.name), ("Father's Name", doc.father_or_spouse_name or ""),
            ("Date of Birth", _d(doc.date_of_birth)), ("Permanent Account Number", doc.id_display),
        )),)  # fmt: skip
    if doc.document_type == "aadhaar":
        return (
            Panel(t, (("", doc.name), ("DOB", _d(doc.date_of_birth)), ("Gender", g), ("", doc.id_display))),
            Panel(f"{t} - ADDRESS", (
                (doc.relation_label, doc.father_or_spouse_name or ""), ("Address", doc.address or ""),
                ("", doc.id_display),
            ), photo=False),
        )  # fmt: skip
    if doc.document_type == "passport":
        given, _, surname = doc.name.rpartition(" ")
        return (
            Panel(t, (
                ("Type", "P"), ("Country Code", "IND"), ("Passport No.", doc.id_display),
                ("Surname", surname), ("Given Name(s)", given), ("Nationality", "INDIAN"),
                ("Sex", (doc.gender or " ")[0].upper()), ("Date of Birth", _d(doc.date_of_birth)),
                ("Date of Issue", _d(doc.issue_date)), ("Date of Expiry", _d(doc.expiry_date)),
            )),
            Panel("PASSPORT - PARTICULARS", (
                ("Name of Father / Legal Guardian", doc.father_or_spouse_name or ""),
                ("Address", doc.address or ""), ("Passport No.", doc.id_display),
            ), photo=False),
        )  # fmt: skip
    if doc.document_type == "driving_licence":
        return (Panel(t, (
            ("DL No.", doc.id_display), ("Name", doc.name),
            (doc.relation_label, doc.father_or_spouse_name or ""), ("Date of Birth", _d(doc.date_of_birth)),
            ("Address", doc.address or ""), ("Issue Date", _d(doc.issue_date)),
            ("Valid Till (NT)", _d(doc.expiry_date)),
        )),)  # fmt: skip
    return (
        Panel(t, (
            ("EPIC No.", doc.id_display), ("Elector's Name", doc.name),
            (doc.relation_label, doc.father_or_spouse_name or ""), ("Gender", g),
            ("Date of Birth", _d(doc.date_of_birth)),
        )),
        Panel(f"{t} - ADDRESS", (("Address", doc.address or ""), ("EPIC No.", doc.id_display)), photo=False),
    )  # fmt: skip


def render(doc: KycDocument, renderer: Renderer) -> bytes:
    return _render_image(doc) if renderer == "image" else _render_digital(doc)


# -------------------------------------------------------------------- image (scan-like)
DPI = 200
CARD_W, CARD_H = 900, 580  # ~ 4.5 x 2.9 in at 200 DPI (enlarged photocopy of an ID card)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _wrap_px(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    width: int,
) -> list[str]:
    words, lines, line = text.split(" "), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _card_image(doc: KycDocument, panel: Panel) -> Image.Image:
    card = Image.new("RGB", (CARD_W, CARD_H), TINTS[doc.document_type])
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, CARD_W - 1, CARD_H - 1), outline=(90, 90, 90), width=3)
    draw.rectangle((0, 0, CARD_W, 62), fill=(70, 90, 120))
    draw.text((20, 31), panel.title, fill=(255, 255, 255), font=_font(30), anchor="lm")

    x = 30
    if panel.photo:
        draw.rectangle((30, 90, 210, 320), fill=(205, 205, 205), outline=(120, 120, 120), width=2)
        draw.text((120, 205), "PHOTO", fill=(110, 110, 110), font=_font(24), anchor="mm")
        x = 240
    y = 88
    label_font, value_font = _font(19), _font(24)
    for label, value in panel.lines:
        if label:
            draw.text((x, y), label, fill=(70, 70, 70), font=label_font)
            y += 22
        for line in _wrap_px(draw, value, value_font, CARD_W - x - 24):
            draw.text((x, y), line, fill=(15, 15, 15), font=value_font)
            y += 28
        y += 6
    return card


def _watermark(img: Image.Image) -> Image.Image:
    layer = Image.new("L", img.size, 0)
    ImageDraw.Draw(layer).text(
        (img.width / 2, img.height / 2), WATERMARK, fill=255, font=_font(48), anchor="mm"
    )
    layer = layer.rotate(28, resample=Image.Resampling.BICUBIC)
    shade = Image.new("RGB", img.size, (200, 60, 60))
    out = img.copy()
    out.paste(shade, (0, 0), layer.point(lambda v: int(v * 0.35)))
    return out


def _render_image(doc: KycDocument) -> bytes:
    page = Image.new("RGB", (int(8.27 * DPI), int(11.69 * DPI)), (250, 250, 248))
    y = 120
    for panel in panels(doc):
        page.paste(_watermark(_card_image(doc, panel)), (140, y))
        y += CARD_H + 80
    ImageDraw.Draw(page).text(
        (140, page.height - 90), SYNTHETIC_NOTICE, fill=(90, 90, 90), font=_font(20)
    )
    buf = BytesIO()
    page.save(
        buf,
        "PDF",
        resolution=float(DPI),
        quality=85,
        title="KYC specimen (synthetic)",
        creationDate=FIXED_PDF_DATE,
        modDate=FIXED_PDF_DATE,
    )
    return buf.getvalue()


# -------------------------------------------------------------------- digital (text layer)
def _render_digital(doc: KycDocument) -> bytes:
    buf = BytesIO()
    c = Canvas(buf, pagesize=A4, invariant=1)
    c.setTitle("KYC specimen (synthetic)")
    page_h = A4[1]
    w, h = 400.0, 250.0
    top = page_h - MARGIN - 20
    for panel in panels(doc):
        r, g, b = (v / 255 for v in TINTS[doc.document_type])
        c.setFillColorRGB(r, g, b)
        c.roundRect(MARGIN, top - h, w, h, 8, stroke=1, fill=1)
        c.setFillColorRGB(0.27, 0.35, 0.47)
        c.rect(MARGIN, top - 26, w, 26, stroke=0, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont(FONT_BOLD, 12)
        c.drawString(MARGIN + 10, top - 18, panel.title)
        x = MARGIN + 12
        if panel.photo:
            c.setFillColorRGB(0.8, 0.8, 0.8)
            c.rect(MARGIN + 12, top - 140, 80, 100, stroke=1, fill=1)
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.setFont(FONT, 9)
            c.drawCentredString(MARGIN + 52, top - 92, "PHOTO")
            x = MARGIN + 104
        y = top - 44
        c.setFillColorRGB(0, 0, 0)
        for label, value in panel.lines:
            if label:
                c.setFont(FONT, 7)
                c.drawString(x, y, label)
                y -= 10
            c.setFont(FONT_BOLD, 9.5)
            for line in wrap_text(value, MARGIN + w - x - 10, FONT_BOLD, 9.5):
                c.drawString(x, y, line)
                y -= 12
            y -= 3
        c.saveState()
        c.setFillColorRGB(0.85, 0.3, 0.3)
        c.setFillAlpha(0.35)
        c.setFont(FONT_BOLD, 18)
        c.translate(MARGIN + w / 2, top - h / 2)
        c.rotate(25)
        c.drawCentredString(0, 0, WATERMARK)
        c.restoreState()
        top -= h + 30
    c.setFillColorRGB(0, 0, 0)
    c.setFont(FONT, 6.5)
    c.drawString(MARGIN, MARGIN, SYNTHETIC_NOTICE)
    c.showPage()
    c.save()
    return buf.getvalue()
