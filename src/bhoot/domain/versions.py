"""The exact versions a job ran with, recorded in every result for reproducibility."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field


class ResolvedVersions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    engine: str  # bhoot.__version__
    schema_ref: str = Field(alias="schema")  # e.g. "bank_statement@3"
    schema_hash: str
    prompt: str | None = None  # e.g. "bank_v2@1"
    prompt_hash: str | None = None
    pack: str | None = None  # e.g. "lending.bank_statement@1"
    migration_id: str | None = None  # None for @draft and ephemeral class schemas
    rule_ids: tuple[str, ...] = ()
    models: Mapping[str, str] = Field(default_factory=dict)  # role -> model id
    ocr_engine: str | None = None  # name + version + profile, or None without OCR
