"""Built-in mode profiles: how much effort (and money) each criticality may get."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from bhoot.domain.modes import Mode, ModeProfile, Tier
from bhoot.domain.schema import Criticality

NORMAL, IMPORTANT, CRITICAL = Criticality.NORMAL, Criticality.IMPORTANT, Criticality.CRITICAL

MODE_PROFILES: Mapping[Mode, ModeProfile] = MappingProxyType({
    Mode.ECONOMY: ModeProfile(
        mode=Mode.ECONOMY,
        max_tier_by_criticality={NORMAL: Tier.REPAIR, IMPORTANT: Tier.REPAIR, CRITICAL: Tier.REPAIR},
        max_repairs_per_field=1,
        max_rounds=3,
        cross_check_threshold=None,
        fuzzy_quote_threshold=88,
    ),
    Mode.BALANCED: ModeProfile(
        mode=Mode.BALANCED,
        max_tier_by_criticality={NORMAL: Tier.REPAIR, IMPORTANT: Tier.STRONG, CRITICAL: Tier.CROSS},
        max_repairs_per_field=2,
        max_rounds=4,
        cross_check_threshold=0.80,
        fuzzy_quote_threshold=90,
    ),
    Mode.MAX_ACCURACY: ModeProfile(
        mode=Mode.MAX_ACCURACY,
        max_tier_by_criticality={NORMAL: Tier.STRONG, IMPORTANT: Tier.STRONG, CRITICAL: Tier.CROSS},
        max_repairs_per_field=2,
        max_rounds=6,
        cross_check_threshold=0.95,
        fuzzy_quote_threshold=94,
    ),
})  # fmt: skip


def profile_for(mode: Mode, overrides: Mapping[str, Any] | None = None) -> ModeProfile:
    """Built-in profile with optional overrides from config (e.g. cross_check_threshold)."""
    base = MODE_PROFILES[mode]
    if not overrides:
        return base
    return ModeProfile.model_validate({**base.model_dump(), **overrides, "mode": mode})
