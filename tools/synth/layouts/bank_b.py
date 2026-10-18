"""Bank B: no grid lines, txn + value date, international grouping, dd-Mon-yyyy dates."""

from __future__ import annotations

from tools.synth.layouts.base import Column, Layout, Style


class BankBLayout(Layout):
    name = "bank_b"
    columns = (
        Column("date", "Txn Date", 52),
        Column("value_date", "Value Date", 52),
        Column("description", "Description", 215),
        Column("debit", "Debit", 65, "right"),
        Column("credit", "Credit", 65, "right"),
        Column("balance", "Balance", 73, "right"),
    )
    style = Style(
        title="Account Statement",
        date_format="%d-%b-%Y",
        grouping="international",
    )
