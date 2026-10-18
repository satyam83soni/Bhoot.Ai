"""JSON Schemas for baseline extraction, mirroring each dataset's expected.json.

Strict-mode friendly: every property is required, absent values are ``null``, no extra keys.
Amounts are asked for as plain decimal strings and dates as ISO strings, so the scorer
compares like with like.
"""

from __future__ import annotations

from typing import Any

_STR = {"type": ["string", "null"]}
_AMOUNT = {"type": ["string", "null"], "description": "Plain decimal, e.g. 125000.50 (no commas)"}
_DATE = {"type": ["string", "null"], "description": "ISO date YYYY-MM-DD"}


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


BANK_STATEMENT = _obj({
    "bank_name": _STR,
    "account_holder": _STR,
    "account_number": _STR,
    "ifsc": _STR,
    "account_type": {"type": ["string", "null"], "enum": ["savings", "current", "od", "cc", "other", None]},
    "currency": _STR,
    "period": _obj({"from_date": _DATE, "to_date": _DATE}),
    "opening_balance": _AMOUNT,
    "closing_balance": _AMOUNT,
    "transactions": {
        "type": "array",
        "items": _obj({
            "date": _DATE,
            "value_date": _DATE,
            "description": _STR,
            "reference": _STR,
            "debit": _AMOUNT,
            "credit": _AMOUNT,
            "balance": _AMOUNT,
        }),
    },
})  # fmt: skip

_PAY_LINE = _obj({"component": _STR, "amount": _AMOUNT})

SALARY_SLIP = _obj({
    "employer_name": _STR,
    "employee_name": _STR,
    "employee_id": _STR,
    "pan": _STR,
    "uan": _STR,
    "pay_period": _obj({"month": {"type": ["integer", "null"]}, "year": {"type": ["integer", "null"]}}),
    "earnings": {"type": "array", "items": _PAY_LINE},
    "deductions": {"type": "array", "items": _PAY_LINE},
    "gross_earnings": _AMOUNT,
    "total_deductions": _AMOUNT,
    "net_pay": _AMOUNT,
    "bank_account_last4": _STR,
})  # fmt: skip

KYC = _obj({
    "document_type": {
        "type": ["string", "null"],
        "enum": ["pan", "aadhaar", "passport", "driving_licence", "voter_id", None],
    },
    "id_number": {"type": ["string", "null"], "description": "As printed, without spaces"},
    "name": _STR,
    "father_or_spouse_name": _STR,
    "date_of_birth": _DATE,
    "gender": {"type": ["string", "null"], "enum": ["male", "female", None]},
    "address": _STR,
    "issue_date": _DATE,
    "expiry_date": _DATE,
})  # fmt: skip

SCHEMAS: dict[str, dict[str, Any]] = {
    "pack:lending.bank_statement": BANK_STATEMENT,
    "pack:lending.salary_slip": SALARY_SLIP,
    "pack:lending.kyc": KYC,
}

INSTRUCTIONS: dict[str, str] = {
    "pack:lending.bank_statement": (
        "Extract the bank statement. Include every transaction row in order; skip opening/closing "
        "balance and carried-forward rows."
    ),
    "pack:lending.salary_slip": "Extract the salary slip, listing every earning and deduction line.",
    "pack:lending.kyc": "Extract the identity document fields.",
}
