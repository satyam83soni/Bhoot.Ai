"""Few-shot examples shown to the model (pure data)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bhoot.domain.document import PageKind


@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    page_kind: PageKind
    output: Mapping[str, Any]  # already in evidence-envelope form
    input_text: str | None = None
    input_image_ref: str | None = None
    section: str | None = None
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if (self.input_text is None) == (self.input_image_ref is None):
            raise ValueError(
                f"example {self.example_id}: give exactly one of input_text or input_image_ref"
            )
