"""Validation: type checks, then field, row and document rules (Chain of Responsibility).

New values (CANDIDATE fields) become VALIDATED or FAILED_CHECKS. Document rules see the whole
current document, so a repair in one place can still break a chain elsewhere: a failing
document rule re-opens VERIFIED fields it implicates.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from bhoot.domain.checks import CheckResult, CheckStatus, Severity
from bhoot.domain.document import Document
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.locales import Locale
from bhoot.domain.rules import Rule, RuleContext, RuleScope
from bhoot.domain.schema import FieldPath
from bhoot.engine.data import assemble

log = logging.getLogger("bhoot.validation")


@dataclass(frozen=True)
class ValidationResult:
    field_states: dict[str, FieldState]
    doc_checks: tuple[CheckResult, ...]


def current_values(field_states: Mapping[str, FieldState]) -> dict[str, Any]:
    """Typed values of every field that has one (what rules and results see)."""
    return {
        path: fs.current.value
        for path, fs in field_states.items()
        if fs.current is not None
        and fs.current.coercion_error is None
        and fs.status is not FieldStatus.ABSENT
    }


class ValidatorChain:
    def __init__(self, rules: Sequence[Rule]) -> None:
        self.field_rules = [r for r in rules if r.scope is RuleScope.FIELD]
        self.row_rules = [r for r in rules if r.scope is RuleScope.ROW]
        self.doc_rules = [r for r in rules if r.scope is RuleScope.DOCUMENT]

    def run(
        self,
        field_states: Mapping[str, FieldState],
        *,
        locale: Locale,
        doc: Document | None = None,
        pack_options: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        values = current_values(field_states)
        ctx = RuleContext(
            values=MappingProxyType(values),
            data=assemble(values),
            locale=locale,
            doc=doc,
            pack_options=MappingProxyType(dict(pack_options or {})),
        )
        open_paths = [p for p, fs in field_states.items() if fs.status is FieldStatus.CANDIDATE]
        type_checks = {p: _type_check(field_states[p]) for p in open_paths}
        type_failed = [FieldPath(p) for p, c in type_checks.items() if c.blocks]

        local: list[CheckResult] = []
        for r in (*self.field_rules, *self.row_rules):
            local.extend(_run(r, ctx))
        document: list[CheckResult] = []
        for r in self.doc_rules:
            if any(_covers(pattern, p) for pattern in r.applies_to for p in type_failed):
                document.append(
                    CheckResult(
                        r.rule_id,
                        CheckStatus.SKIP,
                        r.severity,
                        message="skipped: input values failed type checks",
                    )
                )
            else:
                document.extend(_run(r, ctx))

        out = dict(field_states)
        for path in open_paths:
            fs = field_states[path]
            concrete = fs.path
            checks = (
                type_checks[path],
                *(c for c in (*local, *document) if c.implicates(concrete)),
            )
            blocking = [c for c in checks if c.blocks]
            if blocking:
                reason = "; ".join(c.message or c.check_id for c in blocking)
                out[path] = fs.transition(FieldStatus.FAILED_CHECKS, reason, checks=checks)
            else:
                out[path] = fs.transition(FieldStatus.VALIDATED, "checks passed", checks=checks)

        for path, fs in field_states.items():
            if path in type_checks or fs.status not in (
                FieldStatus.VERIFIED,
                FieldStatus.VALIDATED,
            ):
                continue
            broken = [c for c in document if c.blocks and c.implicates(fs.path)]
            if broken:
                reason = "; ".join(c.message or c.check_id for c in broken)
                out[path] = fs.transition(
                    FieldStatus.FAILED_CHECKS, reason, checks=(*fs.checks, *broken)
                )
        return ValidationResult(out, tuple(document))


def _type_check(fs: FieldState) -> CheckResult:
    spec, cand = fs.spec, fs.current
    check_id = f"type.{spec.py_type}"
    if cand is not None and cand.coercion_error:
        return CheckResult(
            check_id,
            CheckStatus.FAIL,
            paths=(fs.path,),
            message=f"{fs.path}: {cand.coercion_error}",
        )
    if spec.required and (cand is None or cand.value is None):
        return CheckResult(
            "required",
            CheckStatus.FAIL,
            paths=(fs.path,),
            message=f"{fs.path} is required but empty",
        )
    return CheckResult(check_id, CheckStatus.PASS, paths=(fs.path,))


def _run(rule: Rule, ctx: RuleContext) -> list[CheckResult]:
    try:
        return rule.check(ctx)
    except Exception as exc:  # a broken pack rule must not take the job down
        log.exception("rule %s crashed", rule.rule_id)
        return [
            CheckResult(
                rule.rule_id,
                CheckStatus.WARN,
                Severity.WARNING,
                message=f"rule crashed: {type(exc).__name__}: {exc}",
            )
        ]


def _covers(pattern: str, path: FieldPath) -> bool:
    """Does the rule's applies_to pattern cover ``path`` (the path itself or inside it)?"""
    probe = CheckResult("", CheckStatus.FAIL, paths=(FieldPath(pattern),))
    return probe.implicates(path)
