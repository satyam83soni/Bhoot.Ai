from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bhoot.domain.schema import Criticality, FieldPath, FieldSpec, SchemaDefinition


def test_parse_concrete_and_pattern_paths() -> None:
    assert FieldPath("transactions[3].balance").parts() == ("transactions", 3, "balance")
    assert FieldPath("transactions[*].balance").parts() == ("transactions", "*", "balance")
    assert FieldPath("period.from_date").parts() == ("period", "from_date")
    assert FieldPath("transactions[*].balance").is_pattern()
    assert not FieldPath("transactions[3].balance").is_pattern()


@pytest.mark.parametrize(
    "raw", ["", "[0].x", "a..b", "a[x]", "a.[1]", "1abc", "a b", "a[1]b", "a.", ".a", "a[1"]
)
def test_invalid_paths_raise(raw: str) -> None:
    with pytest.raises(ValueError, match="path"):
        FieldPath(raw)


def test_pattern_matching() -> None:
    pattern = FieldPath("transactions[*].balance")
    assert pattern.matches(FieldPath("transactions[12].balance"))
    assert not pattern.matches(FieldPath("transactions[12].debit"))
    assert not pattern.matches(FieldPath("transactions.balance"))
    assert FieldPath("transactions[12].balance").pattern() == pattern


def test_row_helpers() -> None:
    p = FieldPath("transactions[3].balance")
    assert p.parent_row() == FieldPath("transactions[3]")
    assert p.row_index() == 3
    assert p.leaf_name == "balance"
    assert FieldPath("closing_balance").parent_row() is None
    assert FieldPath("transactions[*].balance").with_index(7) == FieldPath(
        "transactions[7].balance"
    )
    assert FieldPath("transactions[2]").child("debit") == FieldPath("transactions[2].debit")


_names = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)
_parts = st.lists(st.one_of(_names, st.integers(0, 999), st.just("*")), max_size=5)


@given(head=_names, tail=_parts)
def test_from_parts_round_trips(head: str, tail: list[str | int]) -> None:
    path = FieldPath.from_parts([head, *tail])
    assert path.parts() == (head, *tail)
    assert FieldPath(path.raw) == path


def _schema() -> SchemaDefinition:
    specs = {
        "closing_balance": FieldSpec(
            FieldPath("closing_balance"), "number", "Decimal", Criticality.CRITICAL
        ),
        "transactions": FieldSpec(FieldPath("transactions"), "array", "list"),
        "transactions[*].balance": FieldSpec(
            FieldPath("transactions[*].balance"), "number", "Decimal", Criticality.CRITICAL
        ),
        "transactions[*].description": FieldSpec(
            FieldPath("transactions[*].description"), "string", "str"
        ),
    }
    return SchemaDefinition("bank_statement", 2, {"type": "object"}, specs, "sha256:abc")


def test_spec_for_concrete_paths() -> None:
    schema = _schema()
    assert schema.spec_for(FieldPath("transactions[5].balance")).criticality is Criticality.CRITICAL
    assert schema.spec_for(FieldPath("closing_balance")).py_type == "Decimal"
    with pytest.raises(KeyError, match="not a field"):
        schema.spec_for(FieldPath("opening_balance"))


def test_leaf_and_critical_specs() -> None:
    schema = _schema()
    assert {s.path.raw for s in schema.leaf_specs()} == {
        "closing_balance", "transactions[*].balance", "transactions[*].description",
    }  # fmt: skip
    assert {s.path.raw for s in schema.critical_paths()} == {
        "closing_balance",
        "transactions[*].balance",
    }
    assert schema.ref == "bank_statement@2"
