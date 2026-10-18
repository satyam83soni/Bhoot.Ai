"""Synthetic KYC documents: PAN, Aadhaar, passport, driving licence, voter ID.

These are plainly designed **specimens** for testing extraction: no emblems, no official
headers, and every rendering carries a "SPECIMEN - NOT A VALID DOCUMENT" watermark. Numbers
follow the public formats (so format rules can be tested) but are randomly generated.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from tools.synth.faker_providers import IndianDataProvider
from tools.synth.ledger import LATEST_PERIOD_END

KycType = Literal["pan", "aadhaar", "passport", "driving_licence", "voter_id"]
KYC_TYPES: tuple[KycType, ...] = ("pan", "aadhaar", "passport", "driving_licence", "voter_id")
Gender = Literal["male", "female"]

STATE_CODES = ("MH", "KA", "TN", "DL", "UP", "GJ", "RJ", "WB", "TS", "KL")


@dataclass(frozen=True)
class KycDocument:
    document_type: KycType
    id_number: str  # canonical: no spaces; masked Aadhaar keeps its X characters
    id_display: str  # exactly as printed (e.g. "2345 6789 0123")
    name: str
    father_or_spouse_name: str | None
    date_of_birth: date | None
    gender: Gender | None
    address: str | None
    issue_date: date | None
    expiry_date: date | None
    masked: bool = False  # Aadhaar printed as XXXX XXXX 1234
    relation_label: str = "Father's Name"


class KycGenerator:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.data = IndianDataProvider(seed)

    def generate(self, document_type: KycType | None = None) -> KycDocument:
        kind = document_type or self.rng.choice(KYC_TYPES)
        return getattr(self, f"_{kind}")()

    # ---------------------------------------------------------------- shared
    def _person(self) -> tuple[str, str, Gender, date]:
        rng, d = self.rng, self.data
        gender: Gender = rng.choice(("male", "female"))
        first = d.fake.first_name_male() if gender == "male" else d.fake.first_name_female()
        surname = d.fake.last_name()
        father = f"{d.fake.first_name_male()} {surname}"
        age_days = rng.randint(19 * 365, 70 * 365)
        dob = LATEST_PERIOD_END - timedelta(days=age_days)
        return f"{first} {surname}", father, gender, dob

    def _letters(self, n: int) -> str:
        return "".join(self.rng.choice(string.ascii_uppercase) for _ in range(n))

    # ---------------------------------------------------------------- types
    def _pan(self) -> KycDocument:
        name, father, _, dob = self._person()
        number = self.data.pan(name.split()[-1][:1])
        return KycDocument(
            "pan", number, number, name.upper(), father.upper(), dob, None, None, None, None
        )

    def _aadhaar(self) -> KycDocument:
        name, father, gender, dob = self._person()
        digits = str(self.rng.randint(2, 9)) + self.data.digits(11)
        masked = self.rng.random() < 0.4
        shown = f"XXXXXXXX{digits[-4:]}" if masked else digits
        display = " ".join(shown[i : i + 4] for i in range(0, 12, 4))
        relation = "S/O" if gender == "male" else self.rng.choice(("D/O", "W/O"))
        return KycDocument(
            "aadhaar", shown, display, name, father, dob, gender, self.data.address(),
            None, None, masked=masked, relation_label=relation,
        )  # fmt: skip

    def _passport(self) -> KycDocument:
        name, father, gender, dob = self._person()
        number = self._letters(1) + self.data.digits(7)
        issued = LATEST_PERIOD_END - timedelta(days=self.rng.randint(30, 9 * 365))
        expiry = issued.replace(year=issued.year + 10) - timedelta(days=1)
        return KycDocument(
            "passport", number, number, name.upper(), father.upper(), dob, gender,
            self.data.address(), issued, expiry,
        )  # fmt: skip

    def _driving_licence(self) -> KycDocument:
        name, father, gender, dob = self._person()
        state = self.rng.choice(STATE_CODES)
        rto = f"{self.rng.randint(1, 50):02d}"
        issued = LATEST_PERIOD_END - timedelta(days=self.rng.randint(30, 15 * 365))
        number = f"{state}{rto}{issued.year}{self.data.digits(7)}"
        display = f"{state}{rto} {issued.year}{number[-7:]}"
        till_year = min(dob.year + 50, issued.year + 20)
        if till_year <= issued.year:  # issued after 50: renewals are valid for 5 years
            till_year = issued.year + 5
        valid_till = date(till_year, dob.month, min(dob.day, 28))
        relation = "S/O" if gender == "male" else self.rng.choice(("D/O", "W/O"))
        return KycDocument(
            "driving_licence", number, display, name.upper(), father.upper(), dob, gender,
            self.data.address(), issued, valid_till, relation_label=relation,
        )  # fmt: skip

    def _voter_id(self) -> KycDocument:
        name, father, gender, dob = self._person()
        number = self._letters(3) + self.data.digits(7)
        relation = (
            "Father's Name" if gender == "male" or self.rng.random() < 0.5 else "Husband's Name"
        )
        return KycDocument(
            "voter_id", number, number, name, father, dob, gender, self.data.address(), None, None,
            relation_label=relation,
        )  # fmt: skip
