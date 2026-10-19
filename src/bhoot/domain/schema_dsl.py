"""Authoring helpers for extraction schemas: ``Schema`` and ``field()``.

A schema is a plain pydantic model. ``field()`` adds what bhoot needs on top: how critical a
field is (critical fields get every check and may be cross-checked), whether a list of rows
shares one quote per row, and a format hint::

    class Transaction(Schema):
        date: date = field(criticality="important")
        debit: Decimal | None = field(criticality="critical")

    class BankStatement(Schema):
        account_number: str = field(criticality="critical")
        transactions: list[Transaction] = field(granularity="row")

Lives in the domain so packs (which may not import the api layer) can define schemas too.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bhoot.domain.schema import Criticality, Granularity


def field(
    *,
    criticality: Criticality | str = Criticality.NORMAL,
    granularity: Granularity | str | None = None,
    fmt: str | None = None,
    description: str | None = None,
    **kwargs: Any,
) -> Any:
    """A pydantic ``Field`` carrying bhoot metadata. Optional fields default to ``None``."""
    extra: dict[str, Any] = dict(kwargs.pop("json_schema_extra", None) or {})
    extra["x-criticality"] = Criticality(criticality).value
    if granularity is not None:
        extra["x-granularity"] = Granularity(granularity).value
    if fmt is not None:
        extra["x-format"] = fmt
    return Field(description=description, json_schema_extra=extra, **kwargs)


class Schema(BaseModel):
    """Base class for extraction schemas (unknown keys are rejected)."""

    model_config = ConfigDict(extra="forbid")
