"""Turn OCR output into page text, words and character boxes in page coordinates.

Words are mapped from prepared-image pixels back to the original image and then to PDF points
(top-left origin), the same space digital pages use, so quote location, bounding boxes and
crops work on scans exactly as on digital PDFs. Words on the same baseline become one text
line, which also joins table cells that the OCR engine read as separate segments.
"""

from __future__ import annotations

from statistics import median

from bhoot.domain.document import BBox, CharBox, Word
from bhoot.ports.ocr import AssembledText, OcrPage, OcrWord, PreparedImage

SAME_LINE = 0.5  # share of the median word height two words' centres may differ by


def _map(transform: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform[:6]
    return a * x + b * y + c, d * x + e * y + f


class OcrTextAssembler:
    def assemble(
        self,
        ocr: OcrPage,
        prepared: PreparedImage,
        *,
        dpi: int,
        page_width_pt: float,
        page_height_pt: float,
    ) -> AssembledText:
        scale = 72.0 / dpi
        placed: list[tuple[OcrWord, BBox]] = []
        for line in ocr.lines:
            for word in line.words:
                b = word.bbox
                corners = [
                    _map(prepared.transform, x, y)
                    for x, y in ((b.x0, b.y0), (b.x1, b.y0), (b.x0, b.y1), (b.x1, b.y1))
                ]
                xs = [min(max(x * scale, 0.0), page_width_pt) for x, _ in corners]
                ys = [min(max(y * scale, 0.0), page_height_pt) for _, y in corners]
                placed.append((word, BBox(min(xs), min(ys), max(xs), max(ys))))
        if not placed:
            return AssembledText("", (), (), 0.0)

        heights = [box.height for _, box in placed if box.height > 0]
        tolerance = SAME_LINE * (median(heights) if heights else 1.0)
        rows: list[list[tuple[OcrWord, BBox]]] = []
        for item in sorted(placed, key=lambda p: (p[1].center[1], p[1].x0)):
            yc = item[1].center[1]
            if rows and abs(yc - _row_center(rows[-1])) <= tolerance:
                rows[-1].append(item)
            else:
                rows.append([item])

        text_parts: list[str] = []
        words: list[Word] = []
        chars: list[CharBox] = []
        pos = 0
        for line_index, row in enumerate(rows):
            row.sort(key=lambda p: p[1].x0)
            if line_index:
                text_parts.append("\n")
                pos += 1
            for k, (word, box) in enumerate(row):
                if k:
                    text_parts.append(" ")
                    pos += 1
                text = word.text
                text_parts.append(text)
                words.append(Word(text, box, word.confidence, line_index))
                step = box.width / max(1, len(text))
                for i, ch in enumerate(text):
                    if not ch.isspace():
                        chars.append(
                            CharBox(
                                ch,
                                BBox(box.x0 + i * step, box.y0, box.x0 + (i + 1) * step, box.y1),
                                pos + i,
                            )
                        )
                pos += len(text)
        mean = sum(w.confidence for w in words) / len(words)
        return AssembledText("".join(text_parts), tuple(words), tuple(chars), mean)


def _row_center(row: list[tuple[OcrWord, BBox]]) -> float:
    return sum(box.center[1] for _, box in row) / len(row)
