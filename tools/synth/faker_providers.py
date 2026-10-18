"""Fictional Indian banking data for synthetic documents.

Every bank, merchant and lender here is invented. Person names and addresses come from
Faker's ``en_IN`` locale, so no real customer data is ever used.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

from faker import Faker


@dataclass(frozen=True)
class FictionalBank:
    name: str
    ifsc_prefix: str  # 4 letters, as in a real IFSC
    code: str  # short code used inside narrations / UTRs


BANKS: tuple[FictionalBank, ...] = (
    FictionalBank("Aster Bank Ltd", "ASTR", "ASTB"),
    FictionalBank("Banyan Bank", "BNYN", "BNYB"),
    FictionalBank("Cedar Co-operative Bank", "CEDR", "CDRB"),
    FictionalBank("Deodar Bank of India", "DEOD", "DEOB"),
    FictionalBank("Elm Small Finance Bank", "ELMB", "ELMS"),
)

MERCHANTS: tuple[str, ...] = (
    "FOODKART",
    "QUICKMART GROCERY",
    "CITYCABS",
    "PAYWELL RECHARGE",
    "BOOKNEST",
    "MEDIPLUS PHARMACY",
    "FUELSTOP",
    "STREAMBOX",
    "HOMEFIX SERVICES",
    "TRAVELNEST",
    "DAILYFRESH DAIRY",
    "STYLEHUB",
)

LENDERS: tuple[str, ...] = (
    "KAVERI FINANCE LTD",
    "NORTHSTAR HOME LOANS",
    "SAHYADRI CREDIT LTD",
    "MEGHNA AUTO FINANCE",
)

UTILITIES: tuple[str, ...] = (
    "CITY POWER DISTRIBUTION",
    "METRO GAS LTD",
    "SKYNET BROADBAND",
)

CITIES: tuple[str, ...] = (
    "MUMBAI",
    "PUNE",
    "BENGALURU",
    "CHENNAI",
    "HYDERABAD",
    "DELHI",
    "KOLKATA",
    "AHMEDABAD",
    "JAIPUR",
    "LUCKNOW",
)

UPI_SUFFIXES: tuple[str, ...] = ("upi", "pay", "bank")

_ALNUM = string.ascii_uppercase + string.digits


class IndianDataProvider:
    """Seeded source of fictional identifiers, names and narration fragments."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.fake = Faker("en_IN")
        self.fake.seed_instance(seed)

    # ---------------------------------------------------------------- people & places
    def person_name(self) -> str:
        return str(self.fake.name())

    def address(self) -> str:
        return str(self.fake.address()).replace("\n", ", ")

    def city(self) -> str:
        return self.rng.choice(CITIES)

    def employer(self) -> str:
        base = str(self.fake.last_name()).upper()
        return f"{base} {self.rng.choice(('TECHNOLOGIES', 'INDUSTRIES', 'SERVICES', 'TEXTILES'))} PVT LTD"

    # ---------------------------------------------------------------- identifiers
    def bank(self) -> FictionalBank:
        return self.rng.choice(BANKS)

    def ifsc(self, bank: FictionalBank) -> str:
        return bank.ifsc_prefix + "0" + "".join(self.rng.choice(_ALNUM) for _ in range(6))

    def account_number(self) -> str:
        length = self.rng.randint(11, 16)
        return str(self.rng.randint(1, 9)) + "".join(
            str(self.rng.randint(0, 9)) for _ in range(length - 1)
        )

    def pan(self, surname_initial: str = "") -> str:
        letters = "".join(self.rng.choice(string.ascii_uppercase) for _ in range(3))
        fifth = (surname_initial or self.rng.choice(string.ascii_uppercase)).upper()[:1]
        digits = "".join(str(self.rng.randint(0, 9)) for _ in range(4))
        return f"{letters}P{fifth}{digits}{self.rng.choice(string.ascii_uppercase)}"

    def digits(self, n: int) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(n))

    def utr(self, bank: FictionalBank) -> str:
        return f"{bank.code}{self.rng.randint(2, 6)}{self.digits(10)}"

    def upi_ref(self) -> str:
        return self.digits(12)

    def cheque_number(self) -> str:
        return self.digits(6)

    def atm_id(self) -> str:
        return "S1" + "".join(self.rng.choice(_ALNUM) for _ in range(6))

    def upi_handle(self, name: str) -> str:
        handle = "".join(ch for ch in name.lower() if ch.isalpha())[:10] or "user"
        return f"{handle}{self.rng.randint(1, 99)}@{self.rng.choice(UPI_SUFFIXES)}"

    def merchant(self) -> str:
        return self.rng.choice(MERCHANTS)

    def lender(self) -> str:
        return self.rng.choice(LENDERS)

    def utility(self) -> str:
        return self.rng.choice(UTILITIES)

    def card_mask(self) -> str:
        return f"XXXXXXXXXXXX{self.digits(4)}"
