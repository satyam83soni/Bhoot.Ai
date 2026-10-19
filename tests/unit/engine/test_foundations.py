"""Engine foundations: work items, registries, events, prices, budget guard, blobs, state serde."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import pytest

from bhoot.adapters.storage import FileBlobStore, MemoryBlobStore
from bhoot.config.models import ModelSpec, ModelTierConfig
from bhoot.domain.candidate import CandidateValue
from bhoot.domain.checks import CheckResult, CheckStatus
from bhoot.domain.cost import Budget, CostEntry, CostReport, Usage
from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import BBox, CharBox, Document, Page, PageKind, Table, TableCell, Word
from bhoot.domain.errors import DuplicateRegistrationError, UnknownStrategyError
from bhoot.domain.evidence import Evidence, EvidenceSource
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.modes import Tier
from bhoot.domain.schema import Criticality, FieldPath, FieldSpec
from bhoot.domain.work import WorkItem, WorkKind, WorkOutcome, WorkStatus
from bhoot.engine.cost import BudgetGuard, CallShape, Price, PriceBook, estimate_usage, estimate_usd
from bhoot.engine.events import EventBus, FieldDecided, NodeStarted
from bhoot.engine.graph.serde import engine_serializer
from bhoot.engine.normalize.conventions import DocConventions
from bhoot.engine.registry import StrategyRegistry


# ---------------------------------------------------------------- work items
def test_work_item_ids_are_readable_and_stable() -> None:
    item_id = WorkItem.make_id(2, WorkKind.REPAIR, "summary", (0, 1, 2), 0)
    assert item_id == "r2-summary-p1-3-repair-0"
    item = WorkItem(item_id, 2, WorkKind.REPAIR, "summary", (0, 1, 2), Tier.REPAIR, "cheap")
    assert item.with_status(WorkStatus.DONE).status is WorkStatus.DONE
    assert item.status is WorkStatus.PENDING


# ---------------------------------------------------------------- registry
def test_registry_finds_rejects_duplicates_and_hints() -> None:
    reg: StrategyRegistry[int] = StrategyRegistry("chunker", {"per_page": 1})
    reg.register("token_budget", 2)
    assert reg.get("per_page") == 1
    assert reg.names() == ["per_page", "token_budget"]
    with pytest.raises(DuplicateRegistrationError):
        reg.register("per_page", 3)
    reg.register("per_page", 3, replace=True)
    with pytest.raises(UnknownStrategyError, match="did you mean 'token_budget'"):
        reg.get("token_budgte")
    copy = reg.copy()
    copy.register("window", 4)
    assert "window" not in reg


# ---------------------------------------------------------------- events
def test_event_bus_filters_by_type_and_survives_a_broken_subscriber(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    seen: list[object] = []
    decided: list[FieldDecided] = []

    def broken(event: object) -> None:
        raise RuntimeError("observer bug")

    bus.subscribe(broken)
    bus.subscribe(seen.append)
    bus.subscribe(decided.append, FieldDecided)
    started = NodeStarted("job", "plan", 0)
    field_decided = FieldDecided("job", "a", DecisionKind.ACCEPT, "rule5", "verified", 0)
    with caplog.at_level(logging.ERROR, logger="bhoot.events"):
        bus.publish(started)
        bus.publish(field_decided)
    assert seen == [started, field_decided]
    assert decided == [field_decided]
    assert "observer bug" in caplog.text


# ---------------------------------------------------------------- prices
def test_prices_come_from_config_only() -> None:
    models = ModelTierConfig(
        cheap=ModelSpec(model="openai/mini", input_cost_per_mtok=0.15, output_cost_per_mtok=0.6),
        strong=ModelSpec(model="anthropic/big"),
    )
    book = PriceBook(models)
    assert book.is_priced("openai/mini")
    assert not book.is_priced("anthropic/big")
    usage = Usage(prompt_tokens=10_000, completion_tokens=2_000, cached_tokens=4_000)
    # cached input is not billed: 6000 x 0.15/M + 2000 x 0.6/M
    assert book.price("openai/mini").cost(usage) == Decimal("0.00210000")
    assert book.price("anthropic/big").cost(usage) == Decimal("0")


def test_estimates_grow_with_pages_images_and_rows() -> None:
    small = CallShape(text_chars=4_000, schema_chars=800, prompt_chars=1_200, leaves=10)
    big = CallShape(
        text_chars=40_000, images=2, schema_chars=800, prompt_chars=1_200, leaves=10,
        expected_rows=120, row_fields=6,
    )  # fmt: skip
    assert estimate_usage(big).prompt_tokens > estimate_usage(small).prompt_tokens
    assert estimate_usage(big).completion_tokens > estimate_usage(small).completion_tokens
    price = Price(Decimal("1"), Decimal("4"))
    assert estimate_usd(price, big) > estimate_usd(price, small) > 0


# ---------------------------------------------------------------- budget guard
async def test_parallel_reservations_never_exceed_the_budget() -> None:
    guard = BudgetGuard()
    guard.open("job")
    budget = Budget(Decimal("1.00"))
    results = await asyncio.gather(
        *(guard.reserve("job", budget, Decimal("0.30")) for _ in range(10))
    )
    granted = [r for r in results if r is not None]
    assert len(granted) == 3
    assert guard.remaining("job", budget) == Decimal("0.10")
    await guard.settle(granted[0], Decimal("0.10"))  # cheaper than estimated
    await guard.release(granted[1])  # never sent
    assert guard.spent("job") == Decimal("0.10")
    assert guard.can_spend("job", budget, Decimal("0.55"))
    assert not guard.can_spend("job", budget, Decimal("0.65"))


async def test_resumed_job_starts_from_what_it_spent() -> None:
    guard = BudgetGuard()
    guard.open("job", already_spent=Decimal("0.24"))
    assert await guard.reserve("job", Budget(Decimal("0.25")), Decimal("0.02")) is None
    assert await guard.reserve("job", Budget(), Decimal("100")) is not None  # unlimited
    guard.close("job")
    assert guard.spent("job") == Decimal("0")


# ---------------------------------------------------------------- blobs
@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_blob_stores_are_content_addressed(kind: str, tmp_path: Path) -> None:
    store = MemoryBlobStore() if kind == "memory" else FileBlobStore(tmp_path)
    key = await store.put(b"\x89PNG page 1", namespace="pages", suffix=".png")
    assert key.startswith("pages/")
    assert key.endswith(".png")
    assert await store.put(b"\x89PNG page 1", namespace="pages", suffix=".png") == key
    assert await store.get(key) == b"\x89PNG page 1"
    assert await store.exists(key)
    missing = "pages/" + "0" * 64 + ".png"
    assert not await store.exists(missing)
    with pytest.raises(KeyError):
        await store.get(missing)
    with pytest.raises(ValueError, match="invalid blob key"):
        await store.get("../../etc/passwd")
    with pytest.raises(ValueError, match="invalid blob namespace"):
        await store.put(b"x", namespace="../up", suffix=".png")
    if kind == "file":
        assert not await asyncio.to_thread(lambda: list(tmp_path.rglob(".tmp-*")))


# ---------------------------------------------------------------- state serializer
def _realistic_state() -> dict[str, object]:
    box = BBox(10, 20, 60, 30)
    page = Page(
        index=0, width=595, height=842, kind=PageKind.TEXT, text="Closing Balance 1,250.00",
        chars=(CharBox("C", BBox(10, 20, 15, 30), 0),), words=(Word("Closing", box),),
        tables=(Table(0, box, (TableCell(0, 0, "Date", box),), 1, 1),), image_ref="pages/abc.png",
    )  # fmt: skip
    doc = Document("doc-1", "sha", "application/pdf", (page,), "statement.pdf")
    path = FieldPath("summary.closing_balance")
    spec = FieldSpec(
        FieldPath("summary.closing_balance"), "number", "Decimal", Criticality.CRITICAL
    )
    candidate = CandidateValue(
        path, "1,250.00", "r0-summary-p1-1-extract-0", 0, Tier.CHEAP, value=Decimal("1250.00"),
        quote="Closing Balance 1,250.00", page=0, model="openai/mini",
        evidence=Evidence(0, "Closing Balance 1,250.00", EvidenceSource.LLM, (0, 24), box, 100.0),
    )  # fmt: skip
    state = FieldState(path, spec).with_candidate(candidate).transition(FieldStatus.VALIDATED)
    state = state.transition(
        FieldStatus.VERIFIED, "grounded", confidence=0.9,
        confidence_signals=MappingProxyType({"exact_quote": 0.35}),
        checks=(CheckResult("type.Decimal", CheckStatus.PASS, paths=(path,)),),
    )  # fmt: skip
    entry = CostEntry(
        "llm_extract", "openai/mini", Tier.CHEAP, "r0", Usage(100, 20), Decimal("0.0001"), 900
    )
    item = WorkItem(
        "r0-summary-p1-1-extract-0", 0, WorkKind.EXTRACT, "summary", (0,), Tier.CHEAP, "cheap"
    )
    return {
        "document": doc,
        "field_states": {path.raw: state},
        "work_items": (item,),
        "work_outcomes": {item.item_id: WorkOutcome(item.item_id, WorkStatus.DONE, "openai/mini")},
        "decisions": (Decision(path, DecisionKind.ACCEPT, "verified", "rule5"),),
        "cost": CostReport.merge(CostReport.of(entry), CostReport.of(entry)),
        "conventions": DocConventions(dayfirst=True, number_grouping="indian"),
        "round": 1,
        "as_of": date(2026, 3, 31),
    }


def test_extraction_state_survives_a_checkpoint_round_trip() -> None:
    serde = engine_serializer()
    state = _realistic_state()
    restored = serde.loads(serde.dumps(state))
    assert restored == state
    field = restored["field_states"]["summary.closing_balance"]  # type: ignore[index]
    assert field.status is FieldStatus.VERIFIED
    assert type(field.spec.criticality) is Criticality
    assert restored["cost"].by_tier == {Tier.CHEAP: Decimal("0.0002")}  # type: ignore[union-attr]
