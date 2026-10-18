"""References to registry items: ``name``, ``name@3``, ``name@latest``, ``name@draft``.

Grammar: ``name[@selector]`` with ``selector := <int> | latest | draft | <label>``. Names are
lowercase identifiers that may contain ``. _ : -`` (``pack:lending.bank_statement``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from bhoot.domain.errors import InvalidRefError

NAME_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,63}$")
LABEL_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class SelectorKind(StrEnum):
    DEFAULT = "default"  # no selector: head / default version
    VERSION = "version"
    LATEST = "latest"
    DRAFT = "draft"  # current code, not yet migrated (development only)
    LABEL = "label"  # server feature; parsed so errors are clear


@dataclass(frozen=True, slots=True)
class Ref:
    name: str
    kind: SelectorKind = SelectorKind.DEFAULT
    version: int | None = None
    label: str | None = None

    @property
    def is_pack(self) -> bool:
        return self.name.startswith("pack:")

    def __str__(self) -> str:
        if self.kind is SelectorKind.VERSION:
            return f"{self.name}@{self.version}"
        if self.kind is SelectorKind.LABEL:
            return f"{self.name}@{self.label}"
        if self.kind in (SelectorKind.LATEST, SelectorKind.DRAFT):
            return f"{self.name}@{self.kind.value}"
        return self.name


def parse_ref(text: str) -> Ref:
    raw = text.strip()
    name, sep, selector = raw.partition("@")
    if not NAME_RE.fullmatch(name):
        raise InvalidRefError(
            f"invalid name {name!r} in {text!r}: use lowercase letters, digits and . _ : - "
            "(2-64 characters, starting with a letter)"
        )
    if not sep:
        return Ref(name)
    if selector == "":
        raise InvalidRefError(
            f"empty selector in {text!r}: use {name}, {name}@latest or {name}@<version>"
        )
    if selector.isdigit():
        version = int(selector)
        if version < 1:
            raise InvalidRefError(f"versions start at 1, got {text!r}")
        return Ref(name, SelectorKind.VERSION, version=version)
    if selector == "latest":
        return Ref(name, SelectorKind.LATEST)
    if selector == "draft":
        return Ref(name, SelectorKind.DRAFT)
    if LABEL_RE.fullmatch(selector):
        return Ref(name, SelectorKind.LABEL, label=selector)
    raise InvalidRefError(f"invalid selector {selector!r} in {text!r}")
