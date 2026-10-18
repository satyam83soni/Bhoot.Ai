"""Ground truth for synthetic documents.

``expected.json`` is produced from the same ledger that was printed, so it is always correct.
It contains plain values only (no evidence envelopes), in the shape of the
``lending.bank_statement`` pack schema (LLD §9.24.1): amounts as decimal strings, dates ISO.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from tools.synth.kyc import KycDocument
from tools.synth.ledger import Ledger
from tools.synth.salary import SalarySlip

DATASET_VERSION = 1


def _amount(value: Decimal | None) -> str | None:
    return None if value is None else f"{value:.2f}"


def bank_statement_expected(ledger: Ledger) -> dict[str, Any]:
    a = ledger.account
    return {
        "bank_name": a.bank.name,
        "account_holder": a.account_holder,
        "account_number": a.account_number,
        "ifsc": a.ifsc,
        "account_type": a.account_type,
        "currency": a.currency,
        "period": {
            "from_date": ledger.period_from.isoformat(),
            "to_date": ledger.period_to.isoformat(),
        },
        "opening_balance": _amount(ledger.opening_balance),
        "closing_balance": _amount(ledger.closing_balance),
        "transactions": [
            {
                "date": t.date.isoformat(),
                "value_date": t.value_date.isoformat(),
                "description": t.description,
                "reference": t.reference,
                "debit": _amount(t.debit),
                "credit": _amount(t.credit),
                "balance": _amount(t.balance),
            }
            for t in ledger.transactions
        ],
    }


def salary_slip_expected(slip: SalarySlip) -> dict[str, Any]:
    """Shape of the ``lending.salary_slip`` pack schema (LLD §9.24.2)."""
    return {
        "employer_name": slip.employer_name,
        "employee_name": slip.employee_name,
        "employee_id": slip.employee_id,
        "pan": slip.pan,
        "uan": slip.uan,
        "pay_period": {"month": slip.pay_month, "year": slip.pay_year},
        "earnings": [
            {"component": e.component, "amount": _amount(e.amount)} for e in slip.earnings
        ],
        "deductions": [
            {"component": d.component, "amount": _amount(d.amount)} for d in slip.deductions
        ],
        "gross_earnings": _amount(slip.gross_earnings),
        "total_deductions": _amount(slip.total_deductions),
        "net_pay": _amount(slip.net_pay),
        "bank_account_last4": slip.bank_account_last4,
    }


def kyc_expected(doc: KycDocument) -> dict[str, Any]:
    """Shape of the ``lending.kyc`` pack schema (LLD §9.24.3). ``id_number`` is what was
    printed (a masked Aadhaar stays masked)."""
    return {
        "document_type": doc.document_type,
        "id_number": doc.id_number,
        "name": doc.name,
        "father_or_spouse_name": doc.father_or_spouse_name,
        "date_of_birth": doc.date_of_birth.isoformat() if doc.date_of_birth else None,
        "gender": doc.gender,
        "address": doc.address,
        "issue_date": doc.issue_date.isoformat() if doc.issue_date else None,
        "expiry_date": doc.expiry_date.isoformat() if doc.expiry_date else None,
    }


def write_case(
    case_dir: Path,
    *,
    document: bytes,
    expected: dict[str, Any],
    meta: dict[str, Any],
    suffix: str = ".pdf",
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"document{suffix}").write_bytes(document)
    _write_lf(case_dir / "expected.json", json.dumps(expected, indent=2, ensure_ascii=False) + "\n")
    _write_lf(case_dir / "meta.json", json.dumps(meta, indent=2, sort_keys=True) + "\n")


def write_manifest(dataset_dir: Path, *, schema: str, cases: list[dict[str, Any]]) -> None:
    manifest = {
        "dataset_version": DATASET_VERSION,
        "schema": schema,
        "locale": {"country": "IN", "currency": "INR", "dayfirst": True},
        "synthetic": True,
        "cases": cases,
    }
    _write_lf(dataset_dir / "manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))


def _write_lf(path: Path, text: str) -> None:
    # LF on every OS, so regenerating a dataset gives byte-identical files everywhere
    path.write_bytes(text.encode("utf-8"))
