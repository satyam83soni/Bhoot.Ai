"""Opt-in training records: verified fields saved for future fine-tuning."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class TrainingRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_version: Literal[1] = 1
    ts: datetime
    job_id: str
    pack: str | None
    schema_ref: str
    path: str
    value: str
    quote: str | None
    page: int | None  # 1-based
    page_kind: str
    tier: int
    model: str | None
    checks: tuple[str, ...] = ()
    doc_sha256: str
    image_ref: str | None = None
