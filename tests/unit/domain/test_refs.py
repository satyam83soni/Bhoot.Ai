from __future__ import annotations

import pytest

from bhoot.domain.errors import InvalidRefError
from bhoot.domain.refs import Ref, SelectorKind, parse_ref


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("bank_statement", Ref("bank_statement")),
        ("bank_statement@3", Ref("bank_statement", SelectorKind.VERSION, version=3)),
        ("loan_doc@latest", Ref("loan_doc", SelectorKind.LATEST)),
        ("loan_doc@draft", Ref("loan_doc", SelectorKind.DRAFT)),
        ("loan_doc@prod", Ref("loan_doc", SelectorKind.LABEL, label="prod")),
        ("pack:lending.bank_statement", Ref("pack:lending.bank_statement")),
        ("  spaced@2 ", Ref("spaced", SelectorKind.VERSION, version=2)),
    ],
)
def test_parse_valid_refs(text: str, expected: Ref) -> None:
    assert parse_ref(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "Bank", "1bank", "b", "bank@", "bank@0", "bank@-1", "bank@Prod", "bank name", "a" * 70],
)
def test_parse_invalid_refs(text: str) -> None:
    with pytest.raises(InvalidRefError):
        parse_ref(text)


@pytest.mark.parametrize(
    "text", ["bank_statement", "bank@3", "bank@latest", "bank@draft", "bank@prod"]
)
def test_str_round_trips(text: str) -> None:
    assert str(parse_ref(text)) == text


def test_pack_refs_are_recognised() -> None:
    assert parse_ref("pack:lending.kyc").is_pack
    assert not parse_ref("kyc").is_pack
