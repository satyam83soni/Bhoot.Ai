"""Results of validation rules and type checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from bhoot.domain.schema import FieldPath


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


class Severity(StrEnum):
    ERROR = "error"  # blocks VERIFIED
    WARNING = "warning"  # recorded, does not block


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str  # e.g. "bank_statement.running_balance"
    status: CheckStatus
    severity: Severity = Severity.ERROR
    paths: tuple[FieldPath, ...] = ()  # fields implicated (targets for repair)
    message: str = ""  # human readable; also fed to repair prompts
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def blocks(self) -> bool:
        """A failed ERROR check prevents the implicated fields from being VERIFIED."""
        return self.status is CheckStatus.FAIL and self.severity is Severity.ERROR

    def implicates(self, path: FieldPath) -> bool:
        """True if ``path`` is one of the check's paths or lies inside one: a check on the row
        ``transactions[4]`` implicates ``transactions[4].balance``."""
        target = path.parts()
        for p in self.paths:
            n = len(p.parts())
            if n <= len(target):
                prefix = FieldPath.from_parts(target[:n])
                if p == prefix or p.matches(prefix):
                    return True
        return False
