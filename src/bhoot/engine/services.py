"""What the engine is given to work with: ports, settings and shared engine objects.

Built once by the bootstrap container and shared by every job. Nodes receive only the parts
they use. Per-job data (the pack, mode, budget, options) travels in ``JobContext``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bhoot.config.models import ModelTierConfig
from bhoot.config.settings import BhootSettings
from bhoot.domain.job import JobSpec
from bhoot.domain.rules import Rule
from bhoot.engine.cost import BudgetGuard, PriceBook
from bhoot.engine.events import EventBus
from bhoot.engine.validation import RuleRegistry
from bhoot.ports.llm import LLMClient
from bhoot.ports.parsing import (
    CropRenderer,
    ImagePreprocessor,
    PageRenderer,
    ParserSelector,
    TableExtractor,
)
from bhoot.ports.storage import BlobStore


@dataclass
class EngineServices:
    settings: BhootSettings
    llm: LLMClient
    blobs: BlobStore
    parsers: ParserSelector
    renderer: PageRenderer
    preprocessor: ImagePreprocessor
    crops: CropRenderer
    tables: TableExtractor | None = None
    events: EventBus = field(default_factory=EventBus)
    budget: BudgetGuard = field(default_factory=BudgetGuard)
    rules: RuleRegistry = field(default_factory=RuleRegistry)
    prices: PriceBook = field(init=False)

    def __post_init__(self) -> None:
        self.prices = PriceBook(self.settings.models)

    @property
    def models(self) -> ModelTierConfig | None:
        return self.settings.models


@dataclass(frozen=True)
class JobContext:
    """One job's fixed inputs. Never stored in graph state (it holds the pack's code and the
    PDF password)."""

    job: JobSpec
    attached_rules: tuple[str, ...] = ()

    @property
    def job_id(self) -> str:
        return self.job.job_id

    def rules(self, registry: RuleRegistry) -> tuple[Rule, ...]:
        return registry.rules_for(self.job.pack, self.attached_rules)

    @property
    def pack_options(self) -> Mapping[str, Any]:
        return self.job.pack_options
