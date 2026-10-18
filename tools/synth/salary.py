"""Consistent synthetic salary slips.

Components are computed so that Σ earnings = gross, Σ deductions = total deductions and
gross - deductions = net pay, exactly as the ``lending.salary_slip`` rules expect. Component
labels vary between slips (``Basic`` / ``Basic Salary``, ``PF`` / ``Provident Fund`` …) the
way real payroll systems differ; the ground truth uses the label that was printed.
"""

from __future__ import annotations

import calendar
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from tools.synth.faker_providers import BANKS, IndianDataProvider
from tools.synth.ledger import EARLIEST_PERIOD_START, LATEST_PERIOD_END

RUPEE = Decimal("1")

# one naming style per slip: (basic, hra, conveyance, medical, special, lta, bonus,
#                            pf, pt, tds, esi, loan)
NAMING_STYLES: tuple[dict[str, str], ...] = (
    {
        "basic": "Basic", "hra": "HRA", "conveyance": "Conveyance", "medical": "Medical",
        "special": "Special Allowance", "lta": "LTA", "bonus": "Bonus",
        "pf": "PF", "pt": "Professional Tax", "tds": "TDS", "esi": "ESI", "loan": "Loan Recovery",
    },
    {
        "basic": "Basic Salary", "hra": "House Rent Allowance", "conveyance": "Conveyance Allowance",
        "medical": "Medical Allowance", "special": "Special Allowance",
        "lta": "Leave Travel Allowance", "bonus": "Performance Bonus",
        "pf": "Provident Fund", "pt": "Professional Tax", "tds": "Income Tax",
        "esi": "ESIC", "loan": "Salary Advance Recovery",
    },
    {
        "basic": "BASIC", "hra": "H.R.A.", "conveyance": "CONV. ALLOW.", "medical": "MED. ALLOW.",
        "special": "SPL. ALLOW.", "lta": "L.T.A.", "bonus": "INCENTIVE",
        "pf": "EPF", "pt": "P.TAX", "tds": "TDS", "esi": "E.S.I.", "loan": "LOAN DED.",
    },
)  # fmt: skip

DESIGNATIONS = (
    "Software Engineer", "Senior Associate", "Sales Executive", "Accounts Officer",
    "Operations Manager", "Customer Support Executive", "Team Lead", "Analyst",
)  # fmt: skip
DEPARTMENTS = ("Engineering", "Sales", "Finance", "Operations", "Support", "HR")


def rupees(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(RUPEE, rounding=ROUND_HALF_UP).quantize(Decimal("0.01"))


@dataclass(frozen=True)
class PayLine:
    component: str
    amount: Decimal


@dataclass(frozen=True)
class SalarySlip:
    employer_name: str
    employer_address: str
    employee_name: str
    employee_id: str
    designation: str
    department: str
    pan: str
    uan: str | None
    date_of_joining: date
    pay_month: int
    pay_year: int
    days_in_month: int
    days_paid: int
    earnings: tuple[PayLine, ...]
    deductions: tuple[PayLine, ...]
    bank_name: str
    bank_account_last4: str

    @property
    def gross_earnings(self) -> Decimal:
        return sum((line.amount for line in self.earnings), Decimal("0.00"))

    @property
    def total_deductions(self) -> Decimal:
        return sum((line.amount for line in self.deductions), Decimal("0.00"))

    @property
    def net_pay(self) -> Decimal:
        return self.gross_earnings - self.total_deductions

    @property
    def lop_days(self) -> int:
        return self.days_in_month - self.days_paid


class SalarySlipGenerator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.data = IndianDataProvider(seed)

    def generate(self) -> SalarySlip:
        rng, d = self.rng, self.data
        names = rng.choice(NAMING_STYLES)

        months = (
            LATEST_PERIOD_END.year - EARLIEST_PERIOD_START.year
        ) * 12 + LATEST_PERIOD_END.month
        index = rng.randint(EARLIEST_PERIOD_START.month - 1, months - 1)
        year, month = EARLIEST_PERIOD_START.year + index // 12, index % 12 + 1
        days_in_month = calendar.monthrange(year, month)[1]
        lop = rng.choice((0, 0, 0, 0, 1, 2))
        factor = Decimal(days_in_month - lop) / Decimal(days_in_month)

        gross_full = Decimal(
            rng.choice(
                (18_500, 24_000, 32_000, 45_000, 62_500, 85_000, 1_20_000, 1_75_000, 2_60_000)
            )
        )
        basic = rupees(gross_full * Decimal(rng.choice(("0.40", "0.45", "0.50"))))
        hra = rupees(basic * Decimal(rng.choice(("0.40", "0.50"))))
        conveyance = Decimal("1600.00") if gross_full > 25_000 else Decimal("0.00")
        medical = Decimal("1250.00") if gross_full > 25_000 else Decimal("0.00")
        lta = rupees(basic * Decimal("0.0833")) if rng.random() < 0.4 else Decimal("0.00")
        special = gross_full - basic - hra - conveyance - medical - lta

        full = [("basic", basic), ("hra", hra), ("conveyance", conveyance), ("medical", medical),
                ("lta", lta), ("special", special)]  # fmt: skip
        earnings = [
            PayLine(names[key], rupees(amount * factor)) for key, amount in full if amount > 0
        ]
        if rng.random() < 0.15:
            earnings.append(
                PayLine(names["bonus"], rupees(rng.choice((2_500, 5_000, 10_000, 15_000))))
            )
        gross = sum((line.amount for line in earnings), Decimal("0.00"))

        paid_basic = earnings[0].amount
        pf_base = min(paid_basic, Decimal(15_000)) if rng.random() < 0.5 else paid_basic
        deductions = [PayLine(names["pf"], rupees(pf_base * Decimal("0.12")))]
        if rng.random() < 0.85:
            deductions.append(PayLine(names["pt"], Decimal("200.00")))
        if gross <= 21_000:
            deductions.append(
                PayLine(names["esi"], rupees(gross * Decimal("0.0075") + Decimal("0.49")))
            )
        annual = gross * 12
        if annual > 7_00_000:
            deductions.append(
                PayLine(names["tds"], rupees((annual - 7_00_000) * Decimal("0.10") / 12))
            )
        if rng.random() < 0.12:
            deductions.append(PayLine(names["loan"], rupees(rng.choice((1_000, 2_500, 5_000)))))

        employee = d.person_name()
        surname_initial = employee.split()[-1][:1]
        joined = date(year, month, 1) - timedelta(days=rng.randint(120, 3_000))
        return SalarySlip(
            employer_name=d.employer(),
            employer_address=d.address(),
            employee_name=employee,
            employee_id=f"EMP{d.digits(5)}",
            designation=rng.choice(DESIGNATIONS),
            department=rng.choice(DEPARTMENTS),
            pan=d.pan(surname_initial),
            uan=f"10{d.digits(10)}" if rng.random() < 0.85 else None,
            date_of_joining=joined,
            pay_month=month,
            pay_year=year,
            days_in_month=days_in_month,
            days_paid=days_in_month - lop,
            earnings=tuple(earnings),
            deductions=tuple(deductions),
            bank_name=rng.choice(BANKS).name,
            bank_account_last4=d.digits(4),
        )
