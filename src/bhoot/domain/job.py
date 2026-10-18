"""Jobs: a request resolved into exact versions, mode, budget and options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any

from bhoot.domain.cost import Budget
from bhoot.domain.modes import Mode, ModeProfile
from bhoot.domain.options import ExtractOptions
from bhoot.domain.pack import DocPack
from bhoot.domain.schema import SchemaDefinition
from bhoot.domain.versions import ResolvedVersions


@dataclass(frozen=True)
class ExtractRequest:
    """What the facade hands to the job factory (no open files, already hashed)."""

    doc_sha256: str
    input_ref: str  # blob store key of the original bytes
    schema: str | SchemaDefinition  # a ref string, or a definition built from a Schema class
    filename: str | None = None
    prompt: str | None = None
    mode: Mode | None = None
    max_cost_usd: Decimal | None = None
    options: ExtractOptions = field(default_factory=ExtractOptions)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    pack: DocPack
    versions: ResolvedVersions
    mode: ModeProfile
    budget: Budget
    options: ExtractOptions
    input_ref: str
    created_at: datetime
    cache_key: str
    pack_options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def without_secrets(self) -> JobSpec:
        """Copy safe to checkpoint: the PDF password is never stored."""
        if self.options.password is None:
            return self
        from dataclasses import replace

        return replace(self, options=self.options.model_copy(update={"password": None}))
