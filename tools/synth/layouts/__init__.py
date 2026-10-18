"""Fictional bank statement layouts (no real bank names or logos)."""

from __future__ import annotations

from tools.synth.layouts.bank_a import BankALayout
from tools.synth.layouts.bank_b import BankBLayout
from tools.synth.layouts.bank_c import BankCLayout
from tools.synth.layouts.bank_d import BankDLayout
from tools.synth.layouts.bank_e import BankELayout
from tools.synth.layouts.base import Layout

LAYOUTS: dict[str, type[Layout]] = {
    cls.name: cls for cls in (BankALayout, BankBLayout, BankCLayout, BankDLayout, BankELayout)
}

__all__ = ["LAYOUTS", "Layout"]
