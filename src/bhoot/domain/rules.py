"""The Rule API: how packs and users write validation rules.

Rules are pure: they read a ``RuleContext`` and return ``CheckResult``s. They live in the
domain so a ``DocPack`` can hold them without depending on the engine; the engine owns the
registry and runs them.

    @rule("loan.emi_positive", scope=RuleScope.FIELD, applies_to=("emi",))
    def emi_positive(ctx):
        return [pass_()] if ctx.values["emi"] > 0 else [fail("emi", "EMI must be positive")]
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from bhoot.domain.checks import CheckResult, CheckStatus, Severity
from bhoot.domain.document import Document
from bhoot.domain.hashing import content_hash
from bhoot.domain.locales import Locale
from bhoot.domain.schema import FieldPath


class RuleScope(StrEnum):
    FIELD = "field"  # one value (formats)
    ROW = "row"  # one array item (debit xor credit)
    DOCUMENT = "document"  # cross-field (opening + credits - debits = closing)


@dataclass(frozen=True)
class RuleContext:
    values: Mapping[str, Any]  # concrete path -> current typed value
    data: Mapping[str, Any]  # assembled nested dict
    locale: Locale
    doc: Document | None = None
    pack_options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def rows(self, array_path: str) -> list[Mapping[str, Any]]:
        node: Any = self.data
        for part in array_path.split("."):
            node = node.get(part) if isinstance(node, Mapping) else None
        return [row for row in node if isinstance(row, Mapping)] if isinstance(node, list) else []


class Rule(ABC):
    rule_id: str
    scope: RuleScope
    applies_to: tuple[str, ...]
    severity: Severity = Severity.ERROR

    @abstractmethod
    def check(self, ctx: RuleContext) -> list[CheckResult]: ...

    def source_hash(self) -> str:
        """Hash of the rule's source code: lets migrations detect a rule edited in place."""
        try:
            source = inspect.getsource(type(self))
        except (OSError, TypeError):
            source = type(self).__qualname__
        return content_hash(source)

    def _finish(self, results: Sequence[CheckResult]) -> list[CheckResult]:
        """Fill in this rule's id and severity on results produced by fail() / pass_()."""
        out = []
        for r in results:
            if r.check_id:  # the rule set its own id: keep the result as is
                out.append(r)
                continue
            severity = Severity.WARNING if r.status is CheckStatus.WARN else self.severity
            out.append(replace(r, check_id=self.rule_id, severity=severity))
        return out


class FunctionRule(Rule):
    """A rule written as a plain function and wrapped by ``@rule``."""

    def __init__(
        self,
        fn: Callable[[RuleContext], list[CheckResult]],
        *,
        rule_id: str,
        scope: RuleScope,
        applies_to: Sequence[str],
        severity: Severity,
    ) -> None:
        self.fn = fn
        self.rule_id = rule_id
        self.scope = scope
        self.applies_to = tuple(applies_to)
        self.severity = severity
        self.__doc__ = fn.__doc__

    def check(self, ctx: RuleContext) -> list[CheckResult]:
        return self._finish(self.fn(ctx))

    def source_hash(self) -> str:
        try:
            source = inspect.getsource(self.fn)
        except (OSError, TypeError):
            source = self.fn.__qualname__
        return content_hash({"id": self.rule_id, "source": source})

    def __repr__(self) -> str:
        return f"<rule {self.rule_id} ({self.scope.value})>"


def rule(
    rule_id: str,
    *,
    scope: RuleScope,
    applies_to: Sequence[str],
    severity: Severity = Severity.ERROR,
) -> Callable[[Callable[[RuleContext], list[CheckResult]]], FunctionRule]:
    """Decorator that turns a function into a Rule. No global registration happens here."""
    if not rule_id or " " in rule_id:
        raise ValueError(f"invalid rule id {rule_id!r}")
    for pattern in applies_to:
        FieldPath(pattern)  # validates the pattern early

    def wrap(fn: Callable[[RuleContext], list[CheckResult]]) -> FunctionRule:
        return FunctionRule(
            fn, rule_id=rule_id, scope=scope, applies_to=applies_to, severity=severity
        )

    return wrap


def fail(paths: str | Sequence[str], message: str, **details: Any) -> CheckResult:
    """A failed check implicating ``paths`` (concrete paths, e.g. 'transactions[4].balance')."""
    items = [paths] if isinstance(paths, str) else list(paths)
    return CheckResult(
        check_id="",
        status=CheckStatus.FAIL,
        paths=tuple(FieldPath(p) for p in items),
        message=message,
        details=MappingProxyType(dict(details)),
    )


def pass_(paths: str | Sequence[str] = ()) -> CheckResult:
    items = [paths] if isinstance(paths, str) else list(paths)
    return CheckResult(
        check_id="", status=CheckStatus.PASS, paths=tuple(FieldPath(p) for p in items)
    )


def warn(paths: str | Sequence[str], message: str, **details: Any) -> CheckResult:
    items = [paths] if isinstance(paths, str) else list(paths)
    return CheckResult(
        check_id="",
        status=CheckStatus.WARN,
        severity=Severity.WARNING,
        paths=tuple(FieldPath(p) for p in items),
        message=message,
        details=MappingProxyType(dict(details)),
    )
