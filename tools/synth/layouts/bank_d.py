"""Bank D: table header only on the first page (later pages must reuse column positions),
no grid, narrow Details column so narrations wrap over several lines, dd.mm.yyyy dates."""

from __future__ import annotations

from tools.synth.layouts.base import Column, Layout, Style


class BankDLayout(Layout):
    name = "bank_d"
    columns = (
        Column("date", "Post Date", 50),
        Column("description", "Details", 150),
        Column("reference", "Ref No", 80),
        Column("debit", "Withdrawals", 75, "right"),
        Column("credit", "Deposits", 75, "right"),
        Column("balance", "Balance", 92, "right"),
    )
    style = Style(
        title="Savings / Current Account Statement",
        date_format="%d.%m.%Y",
        grouping="indian",
        header_every_page=False,
    )
