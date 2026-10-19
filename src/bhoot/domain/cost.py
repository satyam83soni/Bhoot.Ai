"""Token usage, cost accounting and budgets. Money is always ``Decimal``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType

from bhoot.domain.modes import Tier

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    images: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
            self.images + other.images,
        )


@dataclass(frozen=True, slots=True)
class CostEntry:
    node: str
    model: str
    tier: Tier
    work_item_id: str
    usage: Usage
    cost_usd: Decimal
    latency_ms: int
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class CostReport:
    total_usd: Decimal = ZERO
    llm_calls: int = 0
    cache_hits: int = 0
    by_tier: Mapping[Tier, Decimal] = field(default_factory=lambda: MappingProxyType({}))
    fields_by_tier: Mapping[Tier, int] = field(default_factory=lambda: MappingProxyType({}))
    entries: tuple[CostEntry, ...] = ()

    @staticmethod
    def empty() -> CostReport:
        return CostReport()

    @staticmethod
    def of(entry: CostEntry) -> CostReport:
        return CostReport(
            total_usd=entry.cost_usd,
            llm_calls=0 if entry.cache_hit else 1,
            cache_hits=1 if entry.cache_hit else 0,
            by_tier=MappingProxyType({entry.tier: entry.cost_usd}),
            entries=(entry,),
        )

    @staticmethod
    def merge(a: CostReport | None, b: CostReport | None) -> CostReport:
        """Associative merge, usable as a graph state reducer for parallel branches."""
        if a is None:
            return b or CostReport()
        if b is None:
            return a
        by_tier: dict[Tier, Decimal] = dict(a.by_tier)
        for tier, usd in b.by_tier.items():
            by_tier[tier] = by_tier.get(tier, ZERO) + usd
        fields: dict[Tier, int] = dict(a.fields_by_tier)
        for tier, n in b.fields_by_tier.items():
            fields[tier] = fields.get(tier, 0) + n
        return CostReport(
            total_usd=a.total_usd + b.total_usd,
            llm_calls=a.llm_calls + b.llm_calls,
            cache_hits=a.cache_hits + b.cache_hits,
            by_tier=MappingProxyType(dict(sorted(by_tier.items()))),
            fields_by_tier=MappingProxyType(dict(sorted(fields.items()))),
            entries=a.entries + b.entries,
        )


@dataclass(frozen=True, slots=True)
class Budget:
    max_usd: Decimal | None = None  # None = unlimited

    def remaining(self, spent: Decimal) -> Decimal | None:
        return None if self.max_usd is None else max(ZERO, self.max_usd - spent)

    def allows(self, spent: Decimal, estimate: Decimal) -> bool:
        return self.max_usd is None or spent + estimate <= self.max_usd
