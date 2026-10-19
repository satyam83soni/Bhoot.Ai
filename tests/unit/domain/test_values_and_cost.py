from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from bhoot.domain import errors
from bhoot.domain.checks import CheckResult, CheckStatus, Severity
from bhoot.domain.cost import Budget, CostEntry, CostReport, Usage
from bhoot.domain.ids import new_job_id
from bhoot.domain.modes import Mode, ModeProfile, Tier
from bhoot.domain.schema import Criticality, FieldPath


# ---------------------------------------------------------------- cost
def _entry(usd: str, tier: Tier = Tier.CHEAP, cache_hit: bool = False) -> CostEntry:
    return CostEntry("llm_extract", "m", tier, "r0-x", Usage(10, 5), Decimal(usd), 100, cache_hit)


_reports = st.lists(
    st.tuples(st.decimals("0", "1", places=4), st.sampled_from(list(Tier)), st.booleans()),
    max_size=4,
).map(
    lambda items: (
        CostReport.empty()
        if not items
        else _merge_all([CostReport.of(_entry(str(u), t, h)) for u, t, h in items])
    )
)


def _merge_all(reports: list[CostReport]) -> CostReport:
    out = CostReport.empty()
    for r in reports:
        out = CostReport.merge(out, r)
    return out


@given(_reports, _reports, _reports)
def test_cost_merge_is_associative(a: CostReport, b: CostReport, c: CostReport) -> None:
    left = CostReport.merge(CostReport.merge(a, b), c)
    right = CostReport.merge(a, CostReport.merge(b, c))
    assert left.total_usd == right.total_usd
    assert dict(left.by_tier) == dict(right.by_tier)
    assert left.llm_calls == right.llm_calls


def test_cost_merge_identity_and_counts() -> None:
    r = CostReport.merge(
        CostReport.of(_entry("0.01")), CostReport.of(_entry("0.00", cache_hit=True))
    )
    assert CostReport.merge(None, r) is r
    assert CostReport.merge(r, None) is r
    assert r.llm_calls == 1
    assert r.cache_hits == 1
    assert r.total_usd == Decimal("0.01")
    assert Usage(1, 2) + Usage(3, 4, 1, 2) == Usage(4, 6, 1, 2)


def test_budget() -> None:
    assert Budget().allows(Decimal(100), Decimal(100))
    assert Budget().remaining(Decimal(5)) is None
    b = Budget(Decimal("0.10"))
    assert b.allows(Decimal("0.05"), Decimal("0.05"))
    assert not b.allows(Decimal("0.05"), Decimal("0.06"))
    assert b.remaining(Decimal("0.25")) == Decimal(0)


# ---------------------------------------------------------------- modes & checks
def test_mode_profile_validation_and_lookup() -> None:
    profile = ModeProfile(
        mode=Mode.BALANCED,
        max_tier_by_criticality=dict.fromkeys(Criticality, Tier.STRONG),
        max_repairs_per_field=2,
        max_rounds=4,
        cross_check_threshold=0.8,
        fuzzy_quote_threshold=90,
    )
    assert profile.max_tier(Criticality.CRITICAL) is Tier.STRONG
    with pytest.raises(ValidationError):
        ModeProfile(
            mode=Mode.BALANCED, max_tier_by_criticality={}, max_repairs_per_field=-1,
            max_rounds=0, fuzzy_quote_threshold=200,
        )  # fmt: skip


def test_check_result_blocking_and_implication() -> None:
    fail = CheckResult(
        "x", CheckStatus.FAIL, Severity.ERROR, (FieldPath("transactions[*].balance"),)
    )
    warn = CheckResult("y", CheckStatus.FAIL, Severity.WARNING)
    assert fail.blocks
    assert not warn.blocks
    assert not CheckResult("z", CheckStatus.PASS).blocks
    assert fail.implicates(FieldPath("transactions[4].balance"))
    assert not fail.implicates(FieldPath("closing_balance"))


# ---------------------------------------------------------------- ids & errors
def test_job_ids_sort_by_time_and_are_reproducible() -> None:
    a = new_job_id(timestamp_ms=1_000, randomness=bytes(10))
    b = new_job_id(timestamp_ms=2_000, randomness=bytes(10))
    assert a < b
    assert a == new_job_id(timestamp_ms=1_000, randomness=bytes(10))
    assert a.startswith("job_")
    assert len(a) == 4 + 26
    assert new_job_id() != new_job_id()
    with pytest.raises(ValueError, match="10 bytes"):
        new_job_id(randomness=b"short")


def test_error_hierarchy() -> None:
    assert issubclass(errors.SchemaDriftError, errors.RegistryError)
    assert issubclass(errors.ProviderTransientError, errors.LLMError)
    assert issubclass(errors.IllegalTransitionError, errors.EngineError)
    assert all(
        issubclass(cls, errors.BhootError)
        for cls in vars(errors).values()
        if isinstance(cls, type) and issubclass(cls, Exception)
    )
    err = errors.EngineError("boom", node="verify")
    assert err.node == "verify"
