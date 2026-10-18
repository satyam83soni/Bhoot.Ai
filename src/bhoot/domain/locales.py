"""Locale conventions for reading numbers and dates."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Locale(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    country: str = "IN"
    currency: str = "INR"
    dayfirst: bool = True
    number_grouping: Literal["indian", "international", "auto"] = "auto"
