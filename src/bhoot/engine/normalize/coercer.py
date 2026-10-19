"""Turn printed strings into typed values (Decimal, date, int, bool, enum labels, text).

Coercion never raises: a value that can't be read keeps ``coercion_error`` and validation
turns that into a failed ``type.<py_type>`` check, which leads to a repair with a hint.
Custom types plug in through ``COERCERS`` (keyed by ``FieldSpec.py_type``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from bhoot.domain.candidate import CandidateValue
from bhoot.domain.locales import Locale
from bhoot.domain.schema import FieldSpec
from bhoot.engine.normalize.conventions import DocConventions
from bhoot.engine.normalize.dates import parse_date
from bhoot.engine.normalize.numbers import AmountParseError, parse_amount
from bhoot.engine.normalize.text import normalize_for_match
from bhoot.engine.registry import StrategyRegistry

log = logging.getLogger("bhoot.coercer")

ENUM_MATCH_THRESHOLD = 90
_TRUE = frozenset({"yes", "y", "true", "t", "1"})
_FALSE = frozenset({"no", "n", "false", "f", "0"})


@dataclass(frozen=True, slots=True)
class CoerceContext:
    conventions: DocConventions
    locale: Locale


Coercer = Callable[[str, FieldSpec, CoerceContext], Any]


def _decimal(raw: str, spec: FieldSpec, ctx: CoerceContext) -> Any:
    return parse_amount(raw, ctx.conventions.number_grouping).value


def _int(raw: str, spec: FieldSpec, ctx: CoerceContext) -> Any:
    value = _decimal(raw, spec, ctx)
    if value is None:
        return None
    if value != value.to_integral_value():
        raise ValueError(f"{raw!r} is not a whole number")
    return int(value)


def _date(raw: str, spec: FieldSpec, ctx: CoerceContext) -> Any:
    return parse_date(raw, dayfirst=ctx.conventions.dayfirst)


def _bool(raw: str, spec: FieldSpec, ctx: CoerceContext) -> Any:
    text = normalize_for_match(raw)
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{raw!r} is not yes/no")


def _str(raw: str, spec: FieldSpec, ctx: CoerceContext) -> Any:
    return " ".join(raw.split())


def _enum(raw: str, spec: FieldSpec, ctx: CoerceContext) -> Any:
    labels = spec.enum or ()
    text = normalize_for_match(raw)
    for label in labels:
        if normalize_for_match(label) == text:
            return label
    best, score = None, 0.0
    for label in labels:
        s = fuzz.ratio(text, normalize_for_match(label))
        if s > score:
            best, score = label, s
    if best is not None and score >= ENUM_MATCH_THRESHOLD:
        return best
    raise ValueError(f"{raw!r} is not one of {', '.join(labels)}")


COERCERS: StrategyRegistry[Coercer] = StrategyRegistry(
    "coercer",
    {"Decimal": _decimal, "int": _int, "date": _date, "bool": _bool, "str": _str},
)


class TypeCoercer:
    def __init__(self, registry: StrategyRegistry[Coercer] = COERCERS) -> None:
        self.registry = registry

    def coerce(
        self, candidate: CandidateValue, spec: FieldSpec, ctx: CoerceContext
    ) -> CandidateValue:
        if candidate.raw is None:
            return candidate.typed(None)
        raw = str(candidate.raw)
        coercer = self._coercer_for(spec)
        try:
            return candidate.typed(coercer(raw, spec, ctx))
        except (AmountParseError, ValueError) as exc:
            return candidate.failed_coercion(f"{spec.py_type}: {exc}")

    def _coercer_for(self, spec: FieldSpec) -> Coercer:
        if spec.enum:
            return _enum
        if spec.py_type in self.registry:
            return self.registry.get(spec.py_type)
        log.warning("no coercer for type %r (%s); reading it as text", spec.py_type, spec.path)
        return _str
