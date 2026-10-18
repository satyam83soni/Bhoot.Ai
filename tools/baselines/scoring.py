"""Score a predicted extraction against expected.json.

Metrics per case:

* ``field_accuracy``: correct fields / expected non-null fields (row fields included)
* ``wrong_value_rate``: non-null predictions that are wrong / all non-null predictions;
  this is the "confidently wrong" rate, the number the engine must drive towards zero
* ``row_recall`` / ``row_precision``: for arrays (transactions, earnings, deductions)

Values are compared after normalisation: amounts as decimals (commas, Rs., Cr/Dr ignored),
dates in any common format, names case/space-insensitive with a small fuzzy tolerance,
identifiers exactly (spaces removed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from dateutil import parser as dateparser
from rapidfuzz import fuzz

AMOUNT_KEYS = {
    "opening_balance", "closing_balance", "debit", "credit", "balance", "amount",
    "gross_earnings", "total_deductions", "net_pay",
}  # fmt: skip
ID_KEYS = {
    "account_number",
    "ifsc",
    "pan",
    "uan",
    "id_number",
    "employee_id",
    "reference",
    "bank_account_last4",
}
INT_KEYS = {"month", "year"}
ROW_KEYS = {"transactions": ("date", "balance"), "earnings": ("amount",), "deductions": ("amount",)}
TOLERANCE = Decimal("0.005")
FUZZY_MIN = 95


# -------------------------------------------------------------------- normalisers
def to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"(?i)rs\.|\brs\b|\binr\b|₹", "", text)  # currency markers ("Rs." incl. its dot)
    text = re.sub(r"(?i)\s*\b(cr|dr)\.?\s*$", "", text.strip())  # trailing Cr / Dr
    text = re.sub(r"[,\s()]", "", text).rstrip("/-")
    if text in {"", "-"}:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return -number if negative else number


def to_date(value: Any) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return dateparser.parse(text, dayfirst=True).date()
    except (ValueError, OverflowError):
        return None


def _norm_text(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def is_date_key(key: str) -> bool:
    return key.endswith("date") or key in {"date_of_birth", "from_date", "to_date"}


def values_match(key: str, expected: Any, predicted: Any) -> bool:
    if expected is None or predicted is None:
        return expected is None and (predicted is None or str(predicted).strip() == "")
    if key in AMOUNT_KEYS:
        e, p = to_decimal(expected), to_decimal(predicted)
        return e is not None and p is not None and abs(abs(e) - abs(p)) <= TOLERANCE
    if is_date_key(key):
        return to_date(expected) == to_date(predicted) and to_date(expected) is not None
    if key in INT_KEYS:
        try:
            return int(expected) == int(predicted)
        except (TypeError, ValueError):
            return False
    if key in ID_KEYS:
        return str(expected).replace(" ", "").upper() == str(predicted).replace(" ", "").upper()
    e, p = _norm_text(expected), _norm_text(predicted)
    return e == p or fuzz.ratio(e, p) >= FUZZY_MIN


# -------------------------------------------------------------------- scoring
@dataclass
class Score:
    expected_fields: int = 0
    correct: int = 0
    wrong: int = 0  # predicted a non-null value that is incorrect
    missed: int = 0  # expected a value, predicted null
    hallucinated: int = 0  # expected null, predicted a value
    rows_expected: int = 0
    rows_predicted: int = 0
    rows_matched: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def predicted_non_null(self) -> int:
        return self.correct + self.wrong + self.hallucinated

    @property
    def field_accuracy(self) -> float:
        return self.correct / self.expected_fields if self.expected_fields else 1.0

    @property
    def wrong_value_rate(self) -> float:
        bad = self.wrong + self.hallucinated
        return bad / self.predicted_non_null if self.predicted_non_null else 0.0

    @property
    def row_recall(self) -> float:
        return self.rows_matched / self.rows_expected if self.rows_expected else 1.0

    @property
    def row_precision(self) -> float:
        return self.rows_matched / self.rows_predicted if self.rows_predicted else 1.0

    def add(self, other: Score) -> Score:
        for name in ("expected_fields", "correct", "wrong", "missed", "hallucinated",
                     "rows_expected", "rows_predicted", "rows_matched"):  # fmt: skip
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_accuracy": round(self.field_accuracy, 4),
            "wrong_value_rate": round(self.wrong_value_rate, 4),
            "row_recall": round(self.row_recall, 4),
            "row_precision": round(self.row_precision, 4),
            "expected_fields": self.expected_fields,
            "correct": self.correct,
            "wrong": self.wrong,
            "missed": self.missed,
            "hallucinated": self.hallucinated,
            "rows_expected": self.rows_expected,
            "rows_predicted": self.rows_predicted,
            "rows_matched": self.rows_matched,
        }


def _score_value(score: Score, key: str, expected: Any, predicted: Any) -> None:
    present = predicted is not None and str(predicted).strip() != ""
    if expected is None:
        if present:
            score.hallucinated += 1
        return
    score.expected_fields += 1
    if not present:
        score.missed += 1
    elif values_match(key, expected, predicted):
        score.correct += 1
    else:
        score.wrong += 1


def _row_matches(
    keys: tuple[str, ...], expected: dict[str, Any], predicted: dict[str, Any]
) -> bool:
    return all(values_match(k, expected.get(k), predicted.get(k)) for k in keys)


def _score_rows(
    score: Score, name: str, expected_rows: list[Any], predicted_rows: list[Any]
) -> None:
    keys = ROW_KEYS.get(name, ())
    expected_rows = [r for r in expected_rows if isinstance(r, dict)]
    predicted_rows = [r for r in (predicted_rows or []) if isinstance(r, dict)]
    score.rows_expected += len(expected_rows)
    score.rows_predicted += len(predicted_rows)
    unmatched = list(range(len(predicted_rows)))
    for exp in expected_rows:
        match = next((j for j in unmatched if _row_matches(keys, exp, predicted_rows[j])), None)
        if match is None:
            for key, value in exp.items():
                _score_value(score, key, value, None)
            continue
        unmatched.remove(match)
        score.rows_matched += 1
        pred = predicted_rows[match]
        for key, value in exp.items():
            _score_value(score, key, value, pred.get(key))
    if unmatched:
        score.notes.append(f"{name}: {len(unmatched)} predicted rows match no expected row")


def score_case(expected: dict[str, Any], predicted: dict[str, Any] | None) -> Score:
    score = Score()
    predicted = predicted or {}
    for key, value in expected.items():
        pred = predicted.get(key)
        if isinstance(value, list):
            _score_rows(score, key, value, pred if isinstance(pred, list) else [])
        elif isinstance(value, dict):
            nested = pred if isinstance(pred, dict) else {}
            for sub_key, sub_value in value.items():
                _score_value(score, sub_key, sub_value, nested.get(sub_key))
        else:
            _score_value(score, key, value, pred)
    return score
