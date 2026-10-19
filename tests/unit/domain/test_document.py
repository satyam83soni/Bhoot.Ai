from __future__ import annotations

import pytest

from bhoot.domain.document import BBox, CharBox, Document, Page, PageKind, Table, TableCell


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValueError, match="invalid bbox"):
        BBox(10, 0, 5, 5)


def test_bbox_union_and_geometry() -> None:
    a, b = BBox(0, 0, 10, 10), BBox(5, 5, 20, 12)
    u = a.union(b)
    assert u.as_tuple() == (0, 0, 20, 12)
    assert u.width == 20
    assert u.height == 12
    assert a.center == (5, 5)
    assert BBox.union_all([]) is None
    assert BBox.union_all([a, b]) == u


def _page() -> Page:
    text = "Closing Balance 48,250.75"
    chars = tuple(
        CharBox(ch, BBox(10 + i * 5, 100, 14 + i * 5, 110), i) for i, ch in enumerate(text)
    )
    return Page(index=2, width=595, height=842, kind=PageKind.TEXT, text=text, chars=chars)


def test_bbox_for_span_covers_only_the_span() -> None:
    page = _page()
    start = page.text.index("48,250.75")
    box = page.bbox_for_span(start, start + len("48,250.75"))
    assert box is not None
    assert box.x0 == 10 + start * 5
    assert box.x1 == 14 + (start + 8) * 5
    assert page.bbox_for_span(500, 600) is None


def test_page_number_is_one_based() -> None:
    assert _page().number == 3


def test_document_replace_page_keeps_others() -> None:
    pages = tuple(Page(index=i, width=1, height=1) for i in range(3))
    doc = Document(doc_id="d", sha256="x", mime="application/pdf", pages=pages)
    updated = doc.replace_page(Page(index=1, width=1, height=1, kind=PageKind.SCANNED))
    assert updated.page(1).kind is PageKind.SCANNED
    assert updated.page(0) is pages[0]
    assert doc.page(1).kind is PageKind.UNCLASSIFIED  # original unchanged
    assert updated.pages_of_kind(PageKind.SCANNED) == (updated.page(1),)
    assert updated.page_count == 3


def test_table_rows_are_ordered_by_column() -> None:
    cells = (
        TableCell(0, 1, "Narration"), TableCell(0, 0, "Date"),
        TableCell(1, 1, "SALARY"), TableCell(1, 0, "01/08/26"),
    )  # fmt: skip
    table = Table(page_index=0, bbox=None, cells=cells, n_rows=2, n_cols=2)
    assert [c.text for c in table.header_row()] == ["Date", "Narration"]
    assert [[c.text for c in r] for r in table.rows()] == [
        ["Date", "Narration"],
        ["01/08/26", "SALARY"],
    ]
