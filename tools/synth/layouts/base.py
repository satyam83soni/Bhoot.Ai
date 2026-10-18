"""Statement layouts: one ledger, many ways of printing it.

``Layout.render`` is a Template Method: header → (table header → rows)* per page → summary
→ footer. Subclasses only declare columns and a ``Style``; a few override hooks to change
the look. Output is deterministic (``invariant=1``) so the same seed gives the same bytes.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from io import BytesIO
from typing import ClassVar, Literal

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from tools.synth.ledger import Ledger, Transaction

ColumnKey = Literal[
    "date", "value_date", "description", "reference", "debit", "credit", "amount", "drcr", "balance"
]
Align = Literal["left", "right", "center"]

PAGE_W, PAGE_H = A4
MARGIN = 36.0
FONT, FONT_BOLD = "Helvetica", "Helvetica-Bold"
FOOTER_H = 40.0
SUMMARY_H = 80.0
SYNTHETIC_NOTICE = (
    "SYNTHETIC TEST DOCUMENT - fictional bank and data, generated for software testing"
)


@dataclass(frozen=True)
class Column:
    key: ColumnKey
    title: str
    width: float
    align: Align = "left"


@dataclass(frozen=True)
class Style:
    title: str
    date_format: str
    grouping: Literal["indian", "international"]
    balance_suffix: bool = False  # "12,345.00 Cr"
    header_every_page: bool = True
    grid: bool = False
    reverse_chronological: bool = False
    carry_forward_rows: bool = False  # "BALANCE C/F" at page end, "BALANCE B/F" at next page start
    opening_row: bool = True
    currency_label: str = "INR"
    font_size: float = 7.5
    line_height: float = 9.5


@dataclass(frozen=True)
class PrintedRow:
    cells: dict[str, str]
    pseudo: bool = False  # opening / B/F / C/F rows: printed, not part of the truth


# -------------------------------------------------------------------- formatting
def format_amount(value: Decimal, grouping: Literal["indian", "international"]) -> str:
    sign = "-" if value < 0 else ""
    integer, frac = f"{abs(value):.2f}".split(".")
    if grouping == "international":
        grouped = f"{int(integer):,}"
    else:
        head, tail = integer[:-3], integer[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join([*parts, tail]) if parts else tail
    return f"{sign}{grouped}.{frac}"


def wrap_text(text: str, width: float, font: str, size: float) -> list[str]:
    """Greedy wrap that may break after '/', '-' or spaces (narrations rarely have spaces)."""
    tokens: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in "/- ":
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)

    lines: list[str] = []
    line = ""
    for token in tokens:
        if stringWidth(line + token, font, size) <= width:
            line += token
            continue
        if line:
            lines.append(line.rstrip())
            line = ""
        while stringWidth(token, font, size) > width:  # single token wider than the column
            cut = len(token)
            while cut > 1 and stringWidth(token[:cut], font, size) > width:
                cut -= 1
            lines.append(token[:cut])
            token = token[cut:]
        line = token
    if line.strip():
        lines.append(line.rstrip())
    return lines or [""]


# -------------------------------------------------------------------- layout
class Layout(ABC):
    name: ClassVar[str]
    columns: ClassVar[tuple[Column, ...]]
    style: ClassVar[Style]

    # ---------------------------------------------------------------- template method
    def render(self, ledger: Ledger) -> bytes:
        buf = BytesIO()
        canvas = Canvas(buf, pagesize=A4, invariant=1)
        canvas.setTitle(f"{ledger.account.bank.name} statement (synthetic)")
        canvas.setAuthor("bhoot synthetic data generator")

        pages = self.paginate(self.table_rows(ledger))
        for index, rows in enumerate(pages):
            first, last = index == 0, index == len(pages) - 1
            y = (
                self.draw_header(canvas, ledger)
                if first
                else self.draw_continuation_header(canvas, ledger)
            )
            if rows and (first or self.style.header_every_page):
                y = self.draw_table_header(canvas, y)
            for row in rows:
                y = self.draw_row(canvas, row, y)
            if last:
                self.draw_summary(canvas, ledger, y - 12)
            self.draw_footer(canvas, index + 1, len(pages))
            canvas.showPage()
        canvas.save()
        return buf.getvalue()

    # ---------------------------------------------------------------- rows
    def table_rows(self, ledger: Ledger) -> list[PrintedRow]:
        txns = list(ledger.transactions)
        rows = [PrintedRow(self.cells(t)) for t in txns]
        if self.style.reverse_chronological:
            rows.reverse()
        if self.style.opening_row:
            opening = PrintedRow(
                self.pseudo_cells("OPENING BALANCE", ledger.opening_balance, ledger.period_from),
                pseudo=True,
            )
            if self.style.reverse_chronological:
                rows.append(opening)
            else:
                rows.insert(0, opening)
        return rows

    def cells(self, t: Transaction) -> dict[str, str]:
        g = self.style.grouping
        amount = t.debit if t.debit is not None else t.credit
        return {
            "date": self.fmt_date(t.date),
            "value_date": self.fmt_date(t.value_date),
            "description": t.description,
            "reference": t.reference or "",
            "debit": format_amount(t.debit, g) if t.debit is not None else "",
            "credit": format_amount(t.credit, g) if t.credit is not None else "",
            "amount": format_amount(amount, g) if amount is not None else "",
            "drcr": "Dr" if t.debit is not None else "Cr",
            "balance": self.fmt_balance(t.balance),
        }

    def pseudo_cells(self, label: str, balance: Decimal, on: date) -> dict[str, str]:
        return {
            "date": self.fmt_date(on),
            "description": label,
            "balance": self.fmt_balance(balance),
        }

    def fmt_date(self, d: date) -> str:
        return d.strftime(self.style.date_format)

    def fmt_balance(self, value: Decimal) -> str:
        text = format_amount(value, self.style.grouping)
        return f"{text} Cr" if self.style.balance_suffix else text

    # ---------------------------------------------------------------- pagination
    def row_height(self, row: PrintedRow) -> float:
        lines = 1
        for col in self.columns:
            text = row.cells.get(col.key, "")
            if text:
                lines = max(lines, len(wrap_text(text, col.width - 4, FONT, self.style.font_size)))
        return lines * self.style.line_height + 3

    def paginate(self, rows: list[PrintedRow]) -> list[list[PrintedRow]]:
        s = self.style
        header_row_h = s.line_height + 6
        pages: list[list[PrintedRow]] = []
        current: list[PrintedRow] = []
        top = PAGE_H - self.first_page_table_top_offset()
        y = top - header_row_h
        bottom = MARGIN + FOOTER_H
        cf_reserve = s.line_height + 3 if s.carry_forward_rows else 0.0

        for row in rows:
            h = self.row_height(row)
            if y - h < bottom + cf_reserve and current:
                last_balance = current[-1].cells.get("balance", "")
                last_date = current[-1].cells.get("date", "")
                if s.carry_forward_rows:
                    current.append(
                        PrintedRow(
                            {
                                "date": last_date,
                                "description": "BALANCE C/F",
                                "balance": last_balance,
                            },
                            True,
                        )
                    )
                pages.append(current)
                current = []
                y = PAGE_H - self.continuation_table_top_offset()
                if s.header_every_page:
                    y -= header_row_h
                if s.carry_forward_rows:
                    current.append(
                        PrintedRow(
                            {
                                "date": last_date,
                                "description": "BALANCE B/F",
                                "balance": last_balance,
                            },
                            True,
                        )
                    )
                    y -= s.line_height + 3
            current.append(row)
            y -= h
        if y - SUMMARY_H < bottom:  # summary doesn't fit: give it its own page
            pages.append(current)
            current = []
        pages.append(current)
        return pages

    def first_page_table_top_offset(self) -> float:
        return 190.0

    def continuation_table_top_offset(self) -> float:
        return MARGIN + 30.0

    # ---------------------------------------------------------------- drawing hooks
    def draw_header(self, c: Canvas, ledger: Ledger) -> float:
        a = ledger.account
        y = PAGE_H - MARGIN
        c.setFont(FONT_BOLD, 14)
        c.drawString(MARGIN, y - 10, a.bank.name)
        c.setFont(FONT, 8)
        c.drawString(MARGIN, y - 24, f"{a.branch}   IFSC: {a.ifsc}")
        c.setFont(FONT_BOLD, 11)
        c.drawRightString(PAGE_W - MARGIN, y - 10, self.style.title)

        y -= 48
        c.setFont(FONT, 8)
        left = [
            ("Account Holder", a.account_holder.upper()),
            ("Address", a.address),
            ("Account Number", a.account_number),
            ("Account Type", a.account_type.title()),
            ("Currency", a.currency),
        ]
        for label, value in left:
            lines = wrap_text(value, 300, FONT, 8)
            c.setFont(FONT_BOLD, 8)
            c.drawString(MARGIN, y, f"{label}:")
            c.setFont(FONT, 8)
            for i, line in enumerate(lines):
                c.drawString(MARGIN + 80, y - i * 10, line)
            y -= 10 * len(lines) + 2
        c.setFont(FONT_BOLD, 8)
        c.drawString(
            MARGIN,
            y - 4,
            f"Statement period: {self.fmt_date(ledger.period_from)} to {self.fmt_date(ledger.period_to)}",
        )
        return PAGE_H - self.first_page_table_top_offset()

    def draw_continuation_header(self, c: Canvas, ledger: Ledger) -> float:
        c.setFont(FONT_BOLD, 9)
        c.drawString(MARGIN, PAGE_H - MARGIN - 8, ledger.account.bank.name)
        c.setFont(FONT, 8)
        c.drawRightString(
            PAGE_W - MARGIN, PAGE_H - MARGIN - 8, f"Account Number: {ledger.account.account_number}"
        )
        return PAGE_H - self.continuation_table_top_offset()

    def draw_table_header(self, c: Canvas, y: float) -> float:
        s = self.style
        h = s.line_height + 6
        c.setFont(FONT_BOLD, s.font_size)
        x = MARGIN
        for col in self.columns:
            if s.grid:
                c.rect(x, y - h, col.width, h)
            self._draw_cell(c, col.title, x, y - s.line_height - 1, col)
            x += col.width
        if not s.grid:
            c.line(MARGIN, y - h, MARGIN + self.table_width(), y - h)
        return y - h

    def draw_row(self, c: Canvas, row: PrintedRow, y: float) -> float:
        s = self.style
        h = self.row_height(row)
        c.setFont(FONT_BOLD if row.pseudo else FONT, s.font_size)
        x = MARGIN
        for col in self.columns:
            if s.grid:
                c.rect(x, y - h, col.width, h)
            text = row.cells.get(col.key, "")
            for i, line in enumerate(
                wrap_text(text, col.width - 4, FONT, s.font_size) if text else []
            ):
                self._draw_cell(c, line, x, y - s.line_height - i * s.line_height, col)
            x += col.width
        return y - h

    def draw_summary(self, c: Canvas, ledger: Ledger, y: float) -> None:
        g = self.style.grouping
        debits = sum(1 for t in ledger.transactions if t.debit is not None)
        credits = len(ledger.transactions) - debits
        c.setFont(FONT_BOLD, 9)
        c.drawString(MARGIN, y - 10, "Statement Summary")
        c.setFont(FONT, 8)
        items = [
            ("Opening Balance", format_amount(ledger.opening_balance, g)),
            (f"Total Debits ({debits})", format_amount(ledger.total_debits, g)),
            (f"Total Credits ({credits})", format_amount(ledger.total_credits, g)),
            ("Closing Balance", format_amount(ledger.closing_balance, g)),
        ]
        for i, (label, value) in enumerate(items):
            c.drawString(MARGIN, y - 24 - i * 11, label)
            c.drawRightString(MARGIN + 260, y - 24 - i * 11, f"{self.style.currency_label} {value}")

    def draw_footer(self, c: Canvas, page: int, pages: int) -> None:
        c.setFont(FONT, 6.5)
        c.drawString(MARGIN, MARGIN, SYNTHETIC_NOTICE)
        c.drawRightString(PAGE_W - MARGIN, MARGIN, f"Page {page} of {pages}")

    # ---------------------------------------------------------------- helpers
    def table_width(self) -> float:
        return sum(col.width for col in self.columns)

    def _draw_cell(self, c: Canvas, text: str, x: float, y: float, col: Column) -> None:
        if col.align == "right":
            c.drawRightString(x + col.width - 2, y, text)
        elif col.align == "center":
            c.drawCentredString(x + col.width / 2, y, text)
        else:
            c.drawString(x + 2, y, text)
