"""Per-call extraction options (part of every job)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# options that change the result and therefore belong in the cache key
OUTPUT_AFFECTING = ("pages", "force_vision_pages", "max_pages", "pack_options")


class ExtractOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    password: str | None = Field(default=None, repr=False)  # never logged, cached or checkpointed
    max_pages: int | None = Field(default=200, ge=1)
    pages: tuple[int, ...] | None = None  # 1-based page numbers to read
    force_vision_pages: tuple[int, ...] = ()  # 1-based pages to send as images
    use_cache: bool = True
    debug_dir: Path | None = None
    debug_unredacted: bool = False
    training_log: bool | None = None
    timeout_s: float = Field(default=300, gt=0)
    raise_on_failure: bool = False
    include_trace: bool = False
    labels: Mapping[str, str] = Field(default_factory=dict)
    pack_options: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("pages", "force_vision_pages")
    @classmethod
    def _one_based(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None and any(p < 1 for p in value):
            raise ValueError("page numbers are 1-based")
        return value

    def cache_relevant(self) -> dict[str, Any]:
        """The subset of options that changes the extraction result."""
        return {key: getattr(self, key) for key in OUTPUT_AFFECTING}
