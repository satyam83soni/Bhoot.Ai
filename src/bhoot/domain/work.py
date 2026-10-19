"""Kinds and outcomes of units of LLM work (shared by domain values and engine planning)."""

from __future__ import annotations

from enum import StrEnum


class WorkKind(StrEnum):
    EXTRACT = "extract"  # first extraction of a section
    REPAIR = "repair"  # re-ask failing fields with feedback
    CROP = "crop"  # re-read rows/fields from image crops (OCR pages)
    CROSSCHECK = "crosscheck"  # independent answer from a second model


class WorkStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    SKIPPED_BUDGET = "skipped_budget"
    ERROR = "error"
