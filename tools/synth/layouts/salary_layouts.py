"""Salary slip layouts (one page). Three common payroll styles:

* ``salary_side_by_side``: earnings and deductions in one table, side by side
* ``salary_stacked``: earnings table, then deductions table, whole-rupee amounts
* ``salary_boxed``: boxed employee block, "Rs." amounts, attendance line
"""

from __future__ import annotations

import calendar
from abc import ABC, abstractmethod
from decimal import Decimal
from io import BytesIO
from typing import ClassVar

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from tools.synth.layouts.base import (
    FONT,
    FONT_BOLD,
    MARGIN,
    SYNTHETIC_NOTICE,
    format_amount,
    wrap_text,
)
from tools.synth.salary import SalarySlip
from tools.synth.words import rupees_in_words

PAGE_W, PAGE_H = A4


class SalaryLayout(ABC):
    name: ClassVar[str]

    def render(self, slip: SalarySlip) -> bytes:
        buf = BytesIO()
        c = Canvas(buf, pagesize=A4, invariant=1)
        c.setTitle("Salary slip (synthetic)")
        c.setAuthor("bhoot synthetic data generator")
        y = self.draw_company(c, slip)
        y = self.draw_employee(c, slip, y)
        y = self.draw_pay(c, slip, y)
        self.draw_net(c, slip, y)
        c.setFont(FONT, 6.5)
        c.drawString(MARGIN, MARGIN, SYNTHETIC_NOTICE)
        c.showPage()
        c.save()
        return buf.getvalue()

    # ---------------------------------------------------------------- shared pieces
    def amount(self, value: Decimal) -> str:
        return format_amount(value, "indian")

    def period(self, slip: SalarySlip) -> str:
        return f"{calendar.month_name[slip.pay_month]} {slip.pay_year}"

    def draw_company(self, c: Canvas, slip: SalarySlip) -> float:
        y = PAGE_H - MARGIN - 14
        c.setFont(FONT_BOLD, 14)
        c.drawCentredString(PAGE_W / 2, y, slip.employer_name)
        c.setFont(FONT, 8)
        for i, line in enumerate(wrap_text(slip.employer_address, 400, FONT, 8)):
            c.drawCentredString(PAGE_W / 2, y - 14 - i * 10, line)
        c.setFont(FONT_BOLD, 10)
        c.drawCentredString(PAGE_W / 2, y - 44, f"Payslip for the month of {self.period(slip)}")
        return y - 64

    def employee_fields(self, slip: SalarySlip) -> list[tuple[str, str]]:
        fields = [
            ("Employee Name", slip.employee_name),
            ("Employee ID", slip.employee_id),
            ("Designation", slip.designation),
            ("Department", slip.department),
            ("Date of Joining", slip.date_of_joining.strftime("%d/%m/%Y")),
            ("PAN", slip.pan),
            ("Bank", slip.bank_name),
            ("Bank A/c No.", f"XXXXXXXX{slip.bank_account_last4}"),
            ("Days Paid", str(slip.days_paid)),
            ("LOP Days", str(slip.lop_days)),
        ]
        if slip.uan:
            fields.insert(6, ("UAN", slip.uan))
        return fields

    def draw_employee(self, c: Canvas, slip: SalarySlip, y: float) -> float:
        fields = self.employee_fields(slip)
        half = (len(fields) + 1) // 2
        c.setFont(FONT, 8)
        for i, (label, value) in enumerate(fields):
            col, row = divmod(i, half)
            x = MARGIN + col * 265
            yy = y - row * 12
            c.setFont(FONT_BOLD, 8)
            c.drawString(x, yy, f"{label}:")
            c.setFont(FONT, 8)
            c.drawString(x + 90, yy, value)
        return y - half * 12 - 16

    @abstractmethod
    def draw_pay(self, c: Canvas, slip: SalarySlip, y: float) -> float: ...

    def draw_net(self, c: Canvas, slip: SalarySlip, y: float) -> None:
        c.setFont(FONT_BOLD, 10)
        c.drawString(MARGIN, y - 14, f"Net Pay: {self.amount(slip.net_pay)}")
        c.setFont(FONT, 8)
        c.drawString(MARGIN, y - 28, f"({rupees_in_words(slip.net_pay)})")


class SideBySideSalaryLayout(SalaryLayout):
    name = "salary_side_by_side"

    def draw_pay(self, c: Canvas, slip: SalarySlip, y: float) -> float:
        widths = (170, 92, 170, 91)
        xs = [MARGIN + sum(widths[:i]) for i in range(4)]
        row_h = 14
        c.setFont(FONT_BOLD, 8)
        for x, w, title in zip(
            xs, widths, ("Earnings", "Amount", "Deductions", "Amount"), strict=True
        ):
            c.rect(x, y - row_h, w, row_h)
            c.drawString(x + 3, y - 10, title)
        y -= row_h
        c.setFont(FONT, 8)
        for i in range(max(len(slip.earnings), len(slip.deductions))):
            cells = ["", "", "", ""]
            if i < len(slip.earnings):
                cells[0], cells[1] = (
                    slip.earnings[i].component,
                    self.amount(slip.earnings[i].amount),
                )
            if i < len(slip.deductions):
                cells[2], cells[3] = (
                    slip.deductions[i].component,
                    self.amount(slip.deductions[i].amount),
                )
            self._row(c, xs, widths, cells, y, row_h)
            y -= row_h
        c.setFont(FONT_BOLD, 8)
        totals = [
            "Gross Earnings",
            self.amount(slip.gross_earnings),
            "Total Deductions",
            self.amount(slip.total_deductions),
        ]
        self._row(c, xs, widths, totals, y, row_h)
        return y - row_h - 6

    @staticmethod
    def _row(
        c: Canvas, xs: list[float], widths: tuple[int, ...], cells: list[str], y: float, h: float
    ) -> None:
        for j, (x, w, text) in enumerate(zip(xs, widths, cells, strict=True)):
            c.rect(x, y - h, w, h)
            if j % 2:
                c.drawRightString(x + w - 3, y - 10, text)
            else:
                c.drawString(x + 3, y - 10, text)


class StackedSalaryLayout(SalaryLayout):
    name = "salary_stacked"

    def amount(self, value: Decimal) -> str:
        # this payroll system prints whole rupees when there are no paise
        text = format_amount(value, "indian")
        return text[:-3] if text.endswith(".00") else text

    def period(self, slip: SalarySlip) -> str:
        return f"{calendar.month_abbr[slip.pay_month]}-{slip.pay_year}"

    def draw_pay(self, c: Canvas, slip: SalarySlip, y: float) -> float:
        y = self._section(c, "EARNINGS", slip.earnings, "Total Earnings", slip.gross_earnings, y)
        return self._section(
            c, "DEDUCTIONS", slip.deductions, "Total Deductions", slip.total_deductions, y - 8
        )

    def _section(
        self, c: Canvas, title: str, lines: tuple, total_label: str, total: Decimal, y: float
    ) -> float:
        c.setFont(FONT_BOLD, 9)
        c.drawString(MARGIN, y, title)
        c.line(MARGIN, y - 4, MARGIN + 300, y - 4)
        y -= 16
        c.setFont(FONT, 8)
        for line in lines:
            c.drawString(MARGIN + 6, y, line.component)
            c.drawRightString(MARGIN + 300, y, self.amount(line.amount))
            y -= 12
        c.line(MARGIN, y + 6, MARGIN + 300, y + 6)
        c.setFont(FONT_BOLD, 8)
        c.drawString(MARGIN + 6, y - 4, total_label)
        c.drawRightString(MARGIN + 300, y - 4, self.amount(total))
        return y - 20

    def draw_net(self, c: Canvas, slip: SalarySlip, y: float) -> None:
        c.setFont(FONT_BOLD, 10)
        c.drawString(MARGIN, y - 10, "Net Salary Payable")
        c.drawRightString(MARGIN + 300, y - 10, self.amount(slip.net_pay))
        c.setFont(FONT, 8)
        c.drawString(MARGIN, y - 24, f"Amount in words: {rupees_in_words(slip.net_pay)}")


class BoxedSalaryLayout(SalaryLayout):
    name = "salary_boxed"

    def amount(self, value: Decimal) -> str:
        return f"Rs. {format_amount(value, 'indian')}"

    def draw_employee(self, c: Canvas, slip: SalarySlip, y: float) -> float:
        top = y + 10
        bottom = super().draw_employee(c, slip, y)
        c.rect(MARGIN - 4, bottom + 4, PAGE_W - 2 * MARGIN + 8, top - bottom - 4)
        c.setFont(FONT, 8)
        c.drawString(
            MARGIN, bottom - 6, f"Attendance: {slip.days_paid} / {slip.days_in_month} days"
        )
        return bottom - 22

    def draw_pay(self, c: Canvas, slip: SalarySlip, y: float) -> float:
        left_x, right_x, width = MARGIN, MARGIN + 270, 250
        bottom = min(
            self._box(
                c,
                "Earnings",
                slip.earnings,
                "Total Earnings",
                slip.gross_earnings,
                left_x,
                y,
                width,
            ),
            self._box(
                c,
                "Deductions",
                slip.deductions,
                "Total Deductions",
                slip.total_deductions,
                right_x,
                y,
                width,
            ),
        )
        return bottom - 8

    def _box(
        self,
        c: Canvas,
        title: str,
        lines: tuple,
        total_label: str,
        total: Decimal,
        x: float,
        y: float,
        w: float,
    ) -> float:
        c.setFont(FONT_BOLD, 9)
        c.drawString(x + 4, y - 12, title)
        yy = y - 28
        c.setFont(FONT, 8)
        for line in lines:
            c.drawString(x + 4, yy, line.component)
            c.drawRightString(x + w - 4, yy, self.amount(line.amount))
            yy -= 12
        c.setFont(FONT_BOLD, 8)
        c.drawString(x + 4, yy - 4, total_label)
        c.drawRightString(x + w - 4, yy - 4, self.amount(total))
        c.rect(x, yy - 12, w, y - yy + 12)
        return yy - 12


SALARY_LAYOUTS: dict[str, type[SalaryLayout]] = {
    cls.name: cls for cls in (SideBySideSalaryLayout, StackedSalaryLayout, BoxedSalaryLayout)
}
