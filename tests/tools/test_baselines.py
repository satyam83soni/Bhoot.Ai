"""Baseline scoring and runner, tested offline with a fake completion function."""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tools.baselines.base import run_dataset, summarise
from tools.baselines.documents import load_document
from tools.baselines.raw_llm import RawLLMBaseline
from tools.baselines.schemas import SCHEMAS
from tools.baselines.scoring import score_case, to_date, to_decimal, values_match

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"


def _expected(dataset: str, case: str = "case_0001") -> dict[str, Any]:
    return json.loads((EVALS / dataset / case / "expected.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- normalisers
@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("1,25,000.00", Decimal("125000.00")),
        ("Rs. 500/-", Decimal("500")),
        ("₹2,500.50", Decimal("2500.50")),
        ("45,959.91 Cr", Decimal("45959.91")),
        ("(1,234.00)", Decimal("-1234.00")),
        ("", None),
        (None, None),
        ("abc", None),
    ],
)
def test_to_decimal(raw: str | None, value: Decimal | None) -> None:
    assert to_decimal(raw) == value


@pytest.mark.parametrize("raw", ["1990-02-24", "24/02/1990", "24-Feb-1990", "24.02.1990"])
def test_to_date_formats(raw: str) -> None:
    assert to_date(raw).isoformat() == "1990-02-24"  # type: ignore[union-attr]


def test_values_match_rules() -> None:
    assert values_match("closing_balance", "48250.75", "48,250.75")
    assert not values_match("closing_balance", "48250.75", "48250.70")
    assert values_match("name", "Ayaan Sagar", "AYAAN  SAGAR")
    assert values_match("pan", "ABCPD1234E", "ABCPD 1234E")
    assert not values_match("pan", "ABCPD1234E", "ABCPD1234F")
    assert values_match("date_of_birth", "1990-02-24", "24/02/1990")
    assert values_match("month", 9, "9")


# ---------------------------------------------------------------- scoring
@pytest.mark.parametrize("dataset", ["bank_statement", "salary_slip", "kyc"])
def test_perfect_prediction_scores_perfectly(dataset: str) -> None:
    expected = _expected(dataset)
    score = score_case(expected, copy.deepcopy(expected))
    assert score.field_accuracy == 1.0
    assert score.wrong_value_rate == 0.0
    assert score.row_recall == 1.0
    assert score.row_precision == 1.0


def test_missing_row_does_not_shift_later_rows() -> None:
    expected = _expected("bank_statement")
    predicted = copy.deepcopy(expected)
    del predicted["transactions"][3]
    score = score_case(expected, predicted)
    rows = len(expected["transactions"])
    assert score.rows_matched == rows - 1
    assert score.wrong == 0  # later rows still align by (date, balance)
    assert score.missed > 0


def test_wrong_and_hallucinated_values_are_counted() -> None:
    expected = _expected("salary_slip")
    predicted = copy.deepcopy(expected)
    predicted["net_pay"] = "1.00"
    if expected["uan"] is None:
        predicted["uan"] = "100000000000"
    else:
        predicted["employer_name"] = "Completely Different Company"
    score = score_case(expected, predicted)
    assert score.wrong + score.hallucinated == 2
    assert score.wrong_value_rate > 0


def test_none_prediction_misses_everything() -> None:
    expected = _expected("kyc")
    score = score_case(expected, None)
    assert score.correct == 0
    assert score.missed == score.expected_fields


# ---------------------------------------------------------------- runner
def _fake_completion(answer_for: dict[str, dict[str, Any]]) -> Any:
    calls: list[dict[str, Any]] = []

    def complete(**kwargs: Any) -> Any:
        calls.append(kwargs)
        text = json.dumps(answer_for.pop(next(iter(answer_for))))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    complete.calls = calls  # type: ignore[attr-defined]
    return complete


def test_runner_scores_a_dataset_end_to_end() -> None:
    dataset = EVALS / "salary_slip"
    answers = {f"case_{i:04d}": _expected("salary_slip", f"case_{i:04d}") for i in (1, 2, 3)}
    completion = _fake_completion(answers)
    runner = RawLLMBaseline("fake/model", completion=completion)
    results = run_dataset(runner, dataset, limit=3)
    report = summarise(runner, dataset, results)
    assert report["cases"] == 3
    assert report["errors"] == 0
    assert report["metrics"]["field_accuracy"] == 1.0
    request = completion.calls[0]  # type: ignore[attr-defined]
    assert (
        request["response_format"]["json_schema"]["schema"] == SCHEMAS["pack:lending.salary_slip"]
    )
    assert request["temperature"] == 0


def test_scanned_documents_are_sent_as_images() -> None:
    document = load_document(EVALS / "kyc" / "case_0001" / "document.pdf")  # image renderer
    assert document.uses_images
    messages = RawLLMBaseline("fake/model").messages(document, "pack:lending.kyc")
    parts = messages[1]["content"]
    assert any(p["type"] == "image_url" for p in parts)


def test_digital_documents_are_sent_as_text() -> None:
    document = load_document(EVALS / "bank_statement" / "case_0001" / "document.pdf")
    assert not document.uses_images
    assert all(document.pages_text)


def test_failed_call_is_recorded_not_raised() -> None:
    def broken(**_: Any) -> Any:
        raise TimeoutError("provider timed out")

    runner = RawLLMBaseline("fake/model", completion=broken)
    results = run_dataset(runner, EVALS / "kyc", limit=1)
    assert results[0].error is not None
    assert "TimeoutError" in results[0].error
    assert results[0].score.correct == 0
