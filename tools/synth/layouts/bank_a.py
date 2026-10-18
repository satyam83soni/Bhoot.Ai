"""Bank A: gridded table, separate withdrawal/deposit columns, Indian grouping, dd/mm/yy dates,
"BALANCE C/F" / "BALANCE B/F" rows at page breaks (exercises the row de-duplicator)."""

from __future__ import annotations

from tools.synth.layouts.base import Column, Layout, Style


class BankALayout(Layout):
    name = "bank_a"
    columns = (
        Column("date", "Date", 44),
        Column("description", "Narration", 162),
        Column("reference", "Chq./Ref.No.", 78),
        Column("value_date", "Value Dt", 44),
        Column("debit", "Withdrawal Amt.", 64, "right"),
        Column("credit", "Deposit Amt.", 64, "right"),
        Column("balance", "Closing Balance", 66, "right"),
    )
    style = Style(
        title="STATEMENT OF ACCOUNT",
        date_format="%d/%m/%y",
        grouping="indian",
        grid=True,
        carry_forward_rows=True,
    )
