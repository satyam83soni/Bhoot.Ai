"""Bank E: reverse-chronological (newest first), "Rs." currency label, international grouping,
dd Mon yyyy dates, no grid."""

from __future__ import annotations

from tools.synth.layouts.base import Column, Layout, Style


class BankELayout(Layout):
    name = "bank_e"
    columns = (
        Column("date", "Date", 56),
        Column("description", "Transaction Details", 250),
        Column("debit", "Debit", 70, "right"),
        Column("credit", "Credit", 70, "right"),
        Column("balance", "Running Balance", 76, "right"),
    )
    style = Style(
        title="e-Statement",
        date_format="%d %b %Y",
        grouping="international",
        reverse_chronological=True,
        currency_label="Rs.",
    )
