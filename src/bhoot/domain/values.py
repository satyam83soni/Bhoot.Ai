"""Typed results of value parsing, shared by the engine and packs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

Side = Literal["debit", "credit"]
Grouping = Literal["indian", "international", "none"]


@dataclass(frozen=True, slots=True)
class ParsedAmount:
    value: Decimal | None  # None for "-", "", "NIL"
    raw: str
    side: Side | None = None  # from "Dr" / "Cr" markers
    grouping: Grouping = "none"
