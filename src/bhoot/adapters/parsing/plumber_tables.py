"""Tables and words from digital PDFs with pdfplumber (MIT).

Ruled tables come from ``find_tables``; borderless statements produce no table here, so the
engine falls back to building a table from ``extract_words`` (see the word table builder).
pdfplumber already uses top-left coordinates in PDF points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

import pdfplumber

from bhoot.domain.document import BBox, Table, TableCell, Word
from bhoot.domain.errors import CorruptPdfError, EncryptedPdfError
from bhoot.ports.parsing import DocumentInput


def _bbox(raw: Sequence[float] | None) -> BBox | None:
    if raw is None:
        return None
    x0, top, x1, bottom = (float(v) for v in raw)
    return BBox(min(x0, x1), min(top, bottom), max(x0, x1), max(top, bottom))


class PlumberTableExtractor:
    def __init__(self, *, table_settings: Mapping[str, Any] | None = None) -> None:
        self.table_settings = dict(table_settings or {})

    def _open(self, inp: DocumentInput) -> Any:
        try:
            return pdfplumber.open(BytesIO(inp.data), password=inp.password or "")
        except Exception as exc:  # pdfplumber wraps pdfminer errors in several types
            if "password" in str(exc).lower() or type(exc).__name__ == "PDFPasswordIncorrect":
                raise EncryptedPdfError("cannot open the PDF: missing or wrong password") from exc
            raise CorruptPdfError(f"pdfplumber cannot read the PDF: {exc}") from exc

    def extract_tables(
        self, inp: DocumentInput, page_indices: Sequence[int]
    ) -> Mapping[int, tuple[Table, ...]]:
        out: dict[int, tuple[Table, ...]] = {}
        with self._open(inp) as pdf:
            for index in page_indices:
                page = pdf.pages[index]
                tables: list[Table] = []
                for found in page.find_tables(self.table_settings):
                    rows = found.extract()
                    cells: list[TableCell] = []
                    for r, row in enumerate(found.rows):
                        for c, cell_box in enumerate(row.cells):
                            text = rows[r][c] if r < len(rows) and c < len(rows[r]) else None
                            cells.append(TableCell(r, c, (text or "").strip(), _bbox(cell_box)))
                    n_cols = max((len(row.cells) for row in found.rows), default=0)
                    tables.append(
                        Table(index, _bbox(found.bbox), tuple(cells), len(found.rows), n_cols)
                    )
                out[index] = tuple(tables)
        return out

    def extract_words(
        self, inp: DocumentInput, page_indices: Sequence[int]
    ) -> Mapping[int, tuple[Word, ...]]:
        out: dict[int, tuple[Word, ...]] = {}
        with self._open(inp) as pdf:
            for index in page_indices:
                words = pdf.pages[index].extract_words(keep_blank_chars=False, use_text_flow=False)
                out[index] = tuple(
                    Word(w["text"], BBox(w["x0"], w["top"], w["x1"], w["bottom"])) for w in words
                )
        return out
