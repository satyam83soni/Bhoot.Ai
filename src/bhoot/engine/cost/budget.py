"""Budget guard: parallel LLM calls reserve their estimated cost before calling.

Branches of one superstep run at the same time and all see the same spent-so-far in the graph
state, so the state alone cannot stop them from overspending together. The guard keeps the
running total per job: every call reserves its estimate, then settles with the real cost.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal

from bhoot.domain.cost import ZERO, Budget


@dataclass(frozen=True, slots=True)
class Reservation:
    job_id: str
    amount: Decimal


@dataclass(slots=True)
class _JobLedger:
    spent: Decimal = ZERO
    reserved: Decimal = ZERO
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class BudgetGuard:
    def __init__(self) -> None:
        self._jobs: dict[str, _JobLedger] = {}

    def open(self, job_id: str, already_spent: Decimal = ZERO) -> None:
        """Start tracking a job; a resumed job passes what it spent before the pause."""
        self._jobs[job_id] = _JobLedger(spent=already_spent)

    def close(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def spent(self, job_id: str) -> Decimal:
        return self._ledger(job_id).spent

    def remaining(self, job_id: str, budget: Budget) -> Decimal | None:
        ledger = self._ledger(job_id)
        return budget.remaining(ledger.spent + ledger.reserved)

    def can_spend(self, job_id: str, budget: Budget, estimate: Decimal) -> bool:
        """For planning decisions (no reservation): is there room for one more call?"""
        ledger = self._ledger(job_id)
        return budget.allows(ledger.spent + ledger.reserved, estimate)

    async def reserve(self, job_id: str, budget: Budget, estimate: Decimal) -> Reservation | None:
        ledger = self._ledger(job_id)
        async with ledger.lock:
            if not budget.allows(ledger.spent + ledger.reserved, estimate):
                return None
            ledger.reserved += estimate
            return Reservation(job_id, estimate)

    async def settle(self, reservation: Reservation, actual: Decimal) -> None:
        """Replace the reservation with what the call really cost (may be more or less)."""
        ledger = self._ledger(reservation.job_id)
        async with ledger.lock:
            ledger.reserved = max(ZERO, ledger.reserved - reservation.amount)
            ledger.spent += actual

    async def release(self, reservation: Reservation) -> None:
        """The call never happened (error before sending): give the reservation back."""
        await self.settle(reservation, ZERO)

    def _ledger(self, job_id: str) -> _JobLedger:
        ledger = self._jobs.get(job_id)
        if ledger is None:  # a job that was never opened behaves like a fresh one
            ledger = self._jobs[job_id] = _JobLedger()
        return ledger
