"""Merging answers into field states, and planning repairs, escalations, crops and cross-checks."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from decimal import Decimal
from typing import Any

from bhoot.domain.candidate import CandidateRole, CandidateValue
from bhoot.domain.checks import CheckResult, CheckStatus
from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import BBox, Document, Page, PageKind
from bhoot.domain.evidence import Evidence, EvidenceSource
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.modes import Tier
from bhoot.domain.pack import DocPack, SectionSpec
from bhoot.domain.schema import FieldPath
from bhoot.domain.schema_dsl import Schema, field
from bhoot.domain.schema_introspect import introspect
from bhoot.domain.work import WorkItem, WorkKind
from bhoot.engine.merge import Merger, initial_states
from bhoot.engine.planning import RepairPlanner, section_for
from bhoot.engine.prompting import DEFAULT_PROMPT


class Txn(Schema):
    date: dt.date = field(criticality="important")
    description: str = field()
    amount: Decimal = field(criticality="critical")
    balance: Decimal = field(criticality="critical")


class Statement(Schema):
    account: str = field(criticality="critical")
    closing: Decimal = field(criticality="critical")
    note: str | None = None
    transactions: list[Txn] = field(granularity="row")


SCHEMA = introspect(Statement)
PACK = DocPack(
    name="t", version=1, schema=SCHEMA, prompt=DEFAULT_PROMPT,
    sections=(SectionSpec(name="summary", paths=("account", "closing", "note")),
              SectionSpec(name="transactions", paths=("transactions",))),
)  # fmt: skip


def _item(item_id: str, pages: tuple[int, ...], *, kind: WorkKind = WorkKind.EXTRACT,
          window: tuple[int, int] | None = None, round_: int = 0) -> WorkItem:  # fmt: skip
    return WorkItem(item_id, round_, kind, "s", pages, Tier.CHEAP, "cheap", row_window=window)


def _c(path: str, raw: Any, item: str, *, page: int | None = 0, quote: str | None = "q", round_: int = 0,
       tier: Tier = Tier.CHEAP, role: CandidateRole = CandidateRole.PRIMARY) -> CandidateValue:  # fmt: skip
    return CandidateValue(
        FieldPath(path), raw, item, round_, tier, quote=quote, page=page, role=role
    )


def _row(
    i: int, item: str, page: int, desc: str, amount: str, balance: str, round_: int = 0
) -> list[CandidateValue]:
    q = f"{desc} {amount} {balance}"
    return [
        _c(f"transactions[{i}].date", "01/04/2026", item, page=page, quote=q, round_=round_),
        _c(f"transactions[{i}].description", desc, item, page=page, quote=q, round_=round_),
        _c(f"transactions[{i}].amount", amount, item, page=page, quote=q, round_=round_),
        _c(f"transactions[{i}].balance", balance, item, page=page, quote=q, round_=round_),
    ]


def _rows(states: dict[str, FieldState]) -> list[tuple[str, str]]:
    out = []
    i = 0
    while f"transactions[{i}].description" in states:
        out.append((states[f"transactions[{i}].description"].current.raw,  # type: ignore[union-attr]
                    states[f"transactions[{i}].balance"].current.raw))  # type: ignore[union-attr]  # fmt: skip
        i += 1
    return out


# ---------------------------------------------------------------- merge: first round
def test_initial_states_cover_scalars_only() -> None:
    assert set(initial_states(SCHEMA)) == {"account", "closing", "note"}
    assert all(fs.status is FieldStatus.PENDING for fs in initial_states(SCHEMA).values())


def test_first_round_merge_picks_scalars_and_orders_rows() -> None:
    items = [_item("p1", (0,)), _item("p2", (1,)), _item("code", (0,))]
    candidates = [
        _c("closing", "1,250.00", "p2", page=1),
        _c("closing", "1,205.00", "p1", page=0, quote=None),  # no quote: loses
        _c("account", "123", "p1"),
        _c("account", "123", "code", tier=Tier.CODE),  # code wins
        # page 2 answers arrive first; the merge still puts page 1 first
        *_row(0, "p2", 1, "UPI fuel", "50.00", "950.00"),
        *_row(1, "p2", 1, "Closing Balance", "", "950.00"),  # pseudo-row
        *_row(0, "p1", 0, "Opening Balance", "", "1,000.00"),  # pseudo-row
        *_row(1, "p1", 0, "Salary", "100.00", "1,100.00"),
        *_row(2, "p1", 0, "Rent", "150.00", "950.00"),
        _c("closing", "9", "old", round_=-1),  # another round: ignored
    ]  # fmt: skip
    p2_first = [c for c in candidates if c.work_item_id == "p2"]
    others = [c for c in candidates if c.work_item_id != "p2"]
    result = Merger().merge(
        schema=SCHEMA, field_states={}, candidates=[*p2_first, *others], items=items,
        nulls=[FieldPath("note")], round_=0,
    )  # fmt: skip
    states = result.field_states
    assert states["closing"].current.raw == "1,250.00"  # type: ignore[union-attr]
    assert states["account"].current.tier is Tier.CODE  # type: ignore[union-attr]
    assert states["note"].status is FieldStatus.PENDING
    assert result.returned_null == frozenset({"note"})
    assert _rows(states) == [("Salary", "1,100.00"), ("Rent", "950.00"), ("UPI fuel", "950.00")]
    assert states["transactions[2].amount"].spec.criticality.value == "critical"
    assert states["transactions[2].amount"].path == FieldPath("transactions[2].amount")


def test_carried_forward_row_at_the_top_of_a_page_is_dropped() -> None:
    items = [_item("p1", (0,)), _item("p2", (1,))]
    candidates = [*_row(0, "p1", 0, "Rent", "150.00", "950.00"),
                  *_row(0, "p2", 1, "Rent", "150.00", "950.00"),
                  *_row(1, "p2", 1, "Fuel", "50.00", "900.00")]  # fmt: skip
    result = Merger().merge(
        schema=SCHEMA, field_states={}, candidates=candidates, items=items, round_=0
    )
    assert _rows(result.field_states) == [("Rent", "950.00"), ("Fuel", "900.00")]
    assert any("repeated at the top" in w for w in result.warnings)


# ---------------------------------------------------------------- merge: later rounds
def _first_round() -> dict[str, FieldState]:
    items = [_item("p1", (0,))]
    cands = [*_row(0, "p1", 0, "A", "1", "10"), *_row(1, "p1", 0, "B", "2", "20"),
             *_row(2, "p1", 0, "C", "3", "30"), *_row(3, "p1", 0, "D", "4", "40"),
             _c("closing", "40", "p1")]  # fmt: skip
    return (
        Merger()
        .merge(schema=SCHEMA, field_states={}, candidates=cands, items=items, round_=0)
        .field_states
    )


def test_window_repair_replaces_exactly_its_rows() -> None:
    states = _first_round()
    fix = _item("r1", (0,), kind=WorkKind.REPAIR, window=(1, 3), round_=1)
    cands = [
        *_row(1, "r1", 0, "B", "2", "21", round_=1),
        *_row(2, "r1", 0, "C", "3", "31", round_=1),
    ]
    result = Merger().merge(
        schema=SCHEMA, field_states=states, candidates=cands, items=[fix], round_=1
    )
    out = result.field_states
    assert _rows(out) == [("A", "10"), ("B", "21"), ("C", "31"), ("D", "40")]
    assert len(out["transactions[1].balance"].attempts) == 2  # history kept on the same cell
    assert out["transactions[0].balance"] is states["transactions[0].balance"]
    assert result.warnings == ()


def test_window_answer_with_a_missing_row_renumbers_later_rows() -> None:
    states = _first_round()
    fix = _item("r1", (0,), kind=WorkKind.REPAIR, window=(1, 3), round_=1)
    cands = _row(1, "r1", 0, "BC", "5", "30", round_=1)  # two rows came back as one
    result = Merger().merge(
        schema=SCHEMA, field_states=states, candidates=cands, items=[fix], round_=1
    )
    out = result.field_states
    assert _rows(out) == [("A", "10"), ("BC", "30"), ("D", "40")]
    assert out["transactions[2].description"].path == FieldPath("transactions[2].description")
    assert out["transactions[2].description"].current.path == FieldPath(
        "transactions[2].description"
    )  # type: ignore[union-attr]
    assert "transactions[3].description" not in out
    assert "renumbered" in result.warnings[0]


def test_alternatives_and_final_fields_are_never_overwritten() -> None:
    states = _first_round()
    states["account"] = replace(states["account"], final=DecisionKind.ACCEPT)
    cross = _item("x1", (0,), kind=WorkKind.CROSSCHECK, round_=1)
    cands = [
        _c("closing", "41", "x1", round_=1, role=CandidateRole.ALTERNATIVE),
        _c("account", "999", "x1", round_=1),
    ]
    out = (
        Merger()
        .merge(schema=SCHEMA, field_states=states, candidates=cands, items=[cross], round_=1)
        .field_states
    )
    assert out["closing"].current.raw == "40"  # type: ignore[union-attr]
    assert out["closing"].alternatives[-1].raw == "41"
    assert out["account"] is states["account"]


# ---------------------------------------------------------------- repair planning
def _doc() -> Document:
    return Document("d", "s", "application/pdf", (
        Page(0, 595, 842, PageKind.TEXT, "t"), Page(1, 595, 842, PageKind.OCR_TEXT, "o", text_source="ocr"),
        Page(2, 595, 842, PageKind.SCANNED, ""),
    ))  # fmt: skip


def _state(path: str, status: FieldStatus, *, page: int | None = 0, raw: str = "x",
           evidence: Evidence | None = None, attempts: int = 1) -> FieldState:  # fmt: skip
    fs = FieldState(FieldPath(path), SCHEMA.spec_for(FieldPath(path)))
    if status is FieldStatus.PENDING:
        return fs
    for n in range(attempts):
        cand = CandidateValue(FieldPath(path), raw, f"r{n}", n, Tier.CHEAP, quote="q", page=page)
        fs = fs.with_candidate(cand.with_evidence(evidence) if evidence else cand)
    fs = fs.transition(FieldStatus.VALIDATED)
    if status is FieldStatus.UNGROUNDED:
        return fs.transition(FieldStatus.UNGROUNDED, "the quote was not found on page 1")
    return fs.transition(status, "checks failed") if status is not FieldStatus.VALIDATED else fs


def _d(path: str, kind: DecisionKind) -> Decision:
    return Decision(FieldPath(path), kind, "why", "rule8")


def test_repairs_escalations_and_cross_checks_become_work_items() -> None:
    check = CheckResult(
        "t.balance",
        CheckStatus.FAIL,
        paths=(FieldPath("closing"),),
        message="closing does not match the rows",
    )
    states = {
        "closing": replace(
            _state("closing", FieldStatus.FAILED_CHECKS, raw="1,205.00"), checks=(check,)
        ),
        "account": _state("account", FieldStatus.UNGROUNDED),
        "note": _state("note", FieldStatus.PENDING),
    }
    decisions = [
        _d("closing", DecisionKind.REPAIR),
        _d("account", DecisionKind.REPAIR),
        _d("note", DecisionKind.REPAIR),
        Decision(FieldPath("closing"), DecisionKind.ACCEPT, "x", "rule5"),
    ]
    items = RepairPlanner().plan(
        round_=1, decisions=decisions, field_states=states, doc=_doc(), pack=PACK,
        doc_checks=(check,), section_pages={"summary": (0, 1)},
    )  # fmt: skip
    by_paths = {tuple(p.raw for p in i.paths): i for i in items}
    repair = by_paths[("closing", "account")]
    assert (repair.kind, repair.tier, repair.model_role) == (WorkKind.REPAIR, Tier.REPAIR, "cheap")
    assert repair.item_id == "r1-summary-p1-1-repair-0"
    assert "closing does not match the rows" in repair.feedback
    assert "You returned closing = '1,205.00': read it again exactly as printed." in repair.feedback
    assert "account: the quote was not found on page 1." in repair.feedback
    missing = by_paths[("note",)]
    assert missing.page_indices == (0, 1)
    assert missing.feedback == ("No value was returned for note: look for it again.",)

    strong = RepairPlanner().plan(
        round_=2, decisions=[_d("closing", DecisionKind.ESCALATE)],
        field_states={"closing": _state("closing", FieldStatus.FAILED_CHECKS, attempts=2)}, doc=_doc(), pack=PACK,
    )[0]  # fmt: skip
    assert (strong.tier, strong.model_role, strong.use_images) == (Tier.STRONG, "strong", True)

    cross = RepairPlanner().plan(
        round_=2, decisions=[_d("closing", DecisionKind.CROSS_CHECK)],
        field_states={"closing": _state("closing", FieldStatus.VALIDATED)}, doc=_doc(), pack=PACK,
    )[0]  # fmt: skip
    assert (cross.kind, cross.tier, cross.model_role, cross.feedback) == (
        WorkKind.CROSSCHECK,
        Tier.CROSS,
        "cross",
        (),
    )


def test_failing_rows_get_a_window_and_ocr_rows_get_crops() -> None:
    states = _first_round()
    ocr_ev = Evidence(
        1, "C 3 30", EvidenceSource.LLM, (0, 6), BBox(10, 300, 500, 312), ocr_min_confidence=0.6
    )
    bad = states["transactions[2].balance"]
    states["transactions[2].balance"] = replace(
        bad.transition(FieldStatus.VALIDATED).transition(
            FieldStatus.FAILED_CHECKS, "balance break"
        ),
    )
    items = RepairPlanner().plan(
        round_=1, decisions=[_d("transactions[2].balance", DecisionKind.REPAIR)],
        field_states=states, doc=_doc(), pack=PACK,
    )  # fmt: skip
    assert items[0].row_window == (1, 4)
    assert items[0].section == "transactions"

    ocr = states["transactions[2].balance"]
    ocr = replace(ocr, current=ocr.current.with_evidence(ocr_ev))  # type: ignore[union-attr]
    crop = RepairPlanner().plan(
        round_=1, decisions=[_d("transactions[2].balance", DecisionKind.REPAIR)],
        field_states={**states, "transactions[2].balance": ocr}, doc=_doc(), pack=PACK,
    )[0]  # fmt: skip
    assert crop.kind is WorkKind.CROP
    assert crop.use_images
    assert crop.crops[0].bbox == BBox(10, 300, 500, 312)
    assert crop.crops[0].paths == ("transactions[2].balance",)
    assert 1 in crop.page_indices


def test_section_lookup() -> None:
    assert section_for(PACK, FieldPath("transactions[4].amount")) == "transactions"
    assert section_for(PACK, FieldPath("closing")) == "summary"
