"""Consistent synthetic bank ledgers.

A ledger is generated as a list of dated events, then balances are computed in order, so
``balance[i] == balance[i-1] - debit[i] + credit[i]`` always holds. Layouts only change how
the ledger is printed; the ground truth is always the ledger itself.
"""

from __future__ import annotations

import calendar
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from tools.synth.faker_providers import FictionalBank, IndianDataProvider

AccountType = Literal["savings", "current"]

CENT = Decimal("0.01")
MIN_BALANCE = Decimal("500.00")
EARLIEST_PERIOD_START = date(2024, 1, 1)
LATEST_PERIOD_END = date(2026, 9, 30)


def money(value: float | Decimal | int) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class AccountInfo:
    bank: FictionalBank
    branch: str
    ifsc: str
    account_number: str
    account_holder: str
    address: str
    account_type: AccountType
    currency: str = "INR"


@dataclass(frozen=True)
class Transaction:
    date: date
    value_date: date
    description: str
    reference: str | None
    debit: Decimal | None
    credit: Decimal | None
    balance: Decimal
    kind: str  # salary, emi, upi_dr, … (generator metadata, not part of the truth)


@dataclass(frozen=True)
class Ledger:
    account: AccountInfo
    period_from: date
    period_to: date
    opening_balance: Decimal
    transactions: tuple[Transaction, ...]

    @property
    def closing_balance(self) -> Decimal:
        return self.transactions[-1].balance if self.transactions else self.opening_balance

    @property
    def total_debits(self) -> Decimal:
        return sum((t.debit for t in self.transactions if t.debit is not None), Decimal("0.00"))

    @property
    def total_credits(self) -> Decimal:
        return sum((t.credit for t in self.transactions if t.credit is not None), Decimal("0.00"))


@dataclass(order=True)
class _Event:
    date: date
    order: int  # intra-day ordering (credits like salary early, charges late)
    kind: str = field(compare=False)
    description: str = field(compare=False)
    reference: str | None = field(compare=False)
    amount: Decimal = field(compare=False)
    is_credit: bool = field(compare=False)
    value_date: date | None = field(default=None, compare=False)


class LedgerGenerator:
    """Seeded generator of realistic Indian retail bank statements."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.data = IndianDataProvider(seed)

    # ---------------------------------------------------------------- public
    def generate(
        self,
        *,
        months: int | None = None,
        start: date | None = None,
        account_type: AccountType | None = None,
        bank: FictionalBank | None = None,
    ) -> Ledger:
        months = months or self.rng.randint(1, 6)
        if start is None:
            # fixed window (not "today") keeps datasets reproducible and never in the future
            latest_start = _add_months(LATEST_PERIOD_END.replace(day=1), 1 - months)
            span = (
                (latest_start.year - EARLIEST_PERIOD_START.year) * 12
                + latest_start.month
                - EARLIEST_PERIOD_START.month
            )
            start = _add_months(EARLIEST_PERIOD_START, self.rng.randint(0, span))
        period_from = start.replace(day=1)
        period_to = _add_months(period_from, months) - timedelta(days=1)
        account = self._account(
            bank, account_type or self.rng.choice(("savings", "savings", "current"))
        )

        opening = money(self.rng.uniform(5_000, 150_000))
        events = self._events(account, period_from, period_to)
        transactions = self._apply(events, opening)
        return Ledger(account, period_from, period_to, opening, tuple(transactions))

    # ---------------------------------------------------------------- account
    def _account(self, bank: FictionalBank | None, account_type: AccountType) -> AccountInfo:
        bank = bank or self.data.bank()
        return AccountInfo(
            bank=bank,
            branch=f"{self.data.city().title()} Main Branch",
            ifsc=self.data.ifsc(bank),
            account_number=self.data.account_number(),
            account_holder=self.data.person_name(),
            address=self.data.address(),
            account_type=account_type,
        )

    # ---------------------------------------------------------------- events
    def _events(self, account: AccountInfo, start: date, end: date) -> list[_Event]:
        rng, d = self.rng, self.data
        bank = account.bank
        events: list[_Event] = []

        salary = money(
            rng.choice((25_000, 38_500, 52_000, 75_000, 1_10_000, 1_85_000))
            + rng.randint(0, 99) * 10
        )
        employer = d.employer()
        loans = [
            (d.lender(), money(rng.uniform(3_000, 40_000)), d.digits(8))
            for _ in range(rng.randint(0, 2))
        ]
        landlord = d.person_name() if rng.random() < 0.5 else None
        rent = money(rng.choice((8_000, 12_500, 18_000, 25_000, 32_000)))
        utility = d.utility()

        for month_start in _month_starts(start, end):
            last = _month_end(month_start)
            pay_day = _last_working_day(last)
            if start <= pay_day <= end:
                utr = d.utr(bank)
                events.append(
                    _Event(
                        pay_day,
                        0,
                        "salary",
                        f"NEFT CR-{utr}-{employer}-SALARY {pay_day:%b%y}".upper(),
                        utr,
                        salary,
                        True,
                    )
                )
            for lender, emi, loan_id in loans:
                emi_day = month_start.replace(day=rng.choice((5, 7, 10)))
                if start <= emi_day <= end:
                    events.append(
                        _Event(
                            emi_day,
                            5,
                            "emi",
                            f"ACH D- {lender}-{loan_id}",
                            f"ACH{d.digits(9)}",
                            emi,
                            False,
                        )
                    )
            if landlord:
                rent_day = month_start.replace(day=rng.randint(2, 5))
                if start <= rent_day <= end:
                    ref = d.digits(12)
                    events.append(
                        _Event(
                            rent_day,
                            6,
                            "rent",
                            f"IMPS/P2A/{ref}/{landlord.upper()}/RENT",
                            ref,
                            rent,
                            False,
                        )
                    )
            bill_day = month_start.replace(day=rng.randint(10, 20))
            if start <= bill_day <= end:
                events.append(
                    _Event(
                        bill_day,
                        7,
                        "utility",
                        f"BILLPAY/{utility}/{d.digits(10)}",
                        None,
                        money(rng.uniform(400, 4_500)),
                        False,
                    )
                )
            if last.month in (3, 6, 9, 12) and start <= last <= end:
                q_start = _add_months(month_start, -2)
                events.append(
                    _Event(
                        last,
                        1,
                        "interest",
                        f"INT.PD:{q_start:%d-%m-%y} TO {last:%d-%m-%y}",
                        None,
                        money(rng.uniform(80, 900)),
                        True,
                    )
                )
                events.append(
                    _Event(last, 9, "charges", "SMS ALERT CHARGES", None, money("17.70"), False)
                )

        day = start
        while day <= end:
            for _ in range(_poisson(rng, 0.9)):
                merchant = d.merchant()
                ref = d.upi_ref()
                amount = money(min(15_000, rng.lognormvariate(5.6, 0.9)))
                if rng.random() < 0.5:
                    amount = amount.quantize(Decimal("1"))
                events.append(
                    _Event(
                        day,
                        3,
                        "upi_dr",
                        f"UPI/DR/{ref}/{merchant}/{d.upi_handle(merchant)}/PAYMENT",
                        ref,
                        money(amount),
                        False,
                    )
                )
            if rng.random() < 0.06:
                name = d.person_name()
                ref = d.upi_ref()
                events.append(
                    _Event(
                        day,
                        2,
                        "upi_cr",
                        f"UPI/CR/{ref}/{name.upper()}/{d.upi_handle(name)}/UPI",
                        ref,
                        money(rng.choice((500, 1_000, 2_000, 2_500, 5_000))),
                        True,
                    )
                )
            if rng.random() < 0.06:
                events.append(
                    _Event(
                        day,
                        4,
                        "atm",
                        f"ATM WDL-{d.atm_id()}-{d.city()}",
                        None,
                        money(rng.choice((500, 1_000, 2_000, 3_000, 5_000, 10_000))),
                        False,
                    )
                )
            if rng.random() < 0.04:
                merchant = d.merchant()
                events.append(
                    _Event(
                        day,
                        4,
                        "pos",
                        f"POS {d.card_mask()} {merchant}",
                        None,
                        money(rng.uniform(300, 8_000)),
                        False,
                    )
                )
            if rng.random() < 0.02:
                chq = d.cheque_number()
                events.append(
                    _Event(
                        day,
                        2,
                        "cheque",
                        f"CHQ DEP-{chq}-CLG-{d.city()}",
                        chq,
                        money(rng.uniform(5_000, 60_000)),
                        True,
                        value_date=day + timedelta(days=rng.randint(1, 2)),
                    )
                )
            if rng.random() < 0.02:
                utr = d.utr(bank)
                name = d.person_name().upper()
                events.append(
                    _Event(
                        day,
                        4,
                        "neft_dr",
                        f"NEFT DR-{utr}-{name}-TRANSFER",
                        utr,
                        money(rng.uniform(1_000, 25_000)),
                        False,
                    )
                )
            day += timedelta(days=1)

        events.sort()
        return events

    # ---------------------------------------------------------------- balances
    @staticmethod
    def _apply(events: list[_Event], opening: Decimal) -> list[Transaction]:
        balance = opening
        out: list[Transaction] = []
        for ev in events:
            if ev.is_credit:
                balance += ev.amount
                debit, credit = None, ev.amount
            else:
                if balance - ev.amount < MIN_BALANCE:
                    continue  # the account holder couldn't afford it: event never happened
                balance -= ev.amount
                debit, credit = ev.amount, None
            out.append(
                Transaction(
                    date=ev.date,
                    value_date=ev.value_date or ev.date,
                    description=ev.description,
                    reference=ev.reference,
                    debit=debit,
                    credit=credit,
                    balance=balance,
                    kind=ev.kind,
                )
            )
        return out


# -------------------------------------------------------------------- date helpers
def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year, month = d.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _month_end(d: date) -> date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])


def _month_starts(start: date, end: date) -> list[date]:
    out, current = [], start.replace(day=1)
    while current <= end:
        out.append(current)
        current = _add_months(current, 1)
    return out


def _last_working_day(d: date) -> date:
    while d.weekday() >= 5:  # Saturday / Sunday
        d -= timedelta(days=1)
    return d


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's method: fine for the small rates used here
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
