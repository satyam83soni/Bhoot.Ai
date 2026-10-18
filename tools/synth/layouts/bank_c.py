"""Bank C: single Amount column plus a Dr/Cr column, balance printed with a "Cr" suffix,
gridded, Indian grouping, dd-mm-yyyy dates."""

from __future__ import annotations

from tools.synth.layouts.base import Column, Layout, Style


class BankCLayout(Layout):
    name = "bank_c"
    columns = (
        Column("date", "Date", 52),
        Column("description", "Particulars", 250),
        Column("amount", "Amount", 75, "right"),
        Column("drcr", "Dr/Cr", 30, "center"),
        Column("balance", "Balance", 90, "right"),
    )
    style = Style(
        title="Statement of Transactions",
        date_format="%d-%m-%Y",
        grouping="indian",
        balance_suffix=True,
        grid=True,
    )
