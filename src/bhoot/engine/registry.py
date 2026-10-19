"""Name-based strategy registries (selectors, chunkers, merge strategies, coercers, policy rules).

Packs and settings refer to strategies by name, so new ones plug in without core edits, and an
unknown name fails fast with a hint instead of deep inside a job.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

from bhoot.domain.errors import DuplicateRegistrationError, UnknownStrategyError

V = TypeVar("V")


class StrategyRegistry(Generic[V]):
    def __init__(self, kind: str, items: Mapping[str, V] | None = None) -> None:
        self.kind = kind
        self._items: dict[str, V] = {}
        for name, item in (items or {}).items():
            self.register(name, item)

    def register(self, name: str, item: V, *, replace: bool = False) -> None:
        if not name:
            raise ValueError(f"{self.kind} name must not be empty")
        if name in self._items and not replace:
            raise DuplicateRegistrationError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item

    def get(self, name: str) -> V:
        try:
            return self._items[name]
        except KeyError:
            close = difflib.get_close_matches(name, self._items, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise UnknownStrategyError(
                f"unknown {self.kind} {name!r}{hint} (known: {', '.join(sorted(self._items))})"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def names(self) -> list[str]:
        return sorted(self._items)

    def copy(self) -> StrategyRegistry[V]:
        """A copy a job or test can extend without touching the shared defaults."""
        return StrategyRegistry(self.kind, self._items)
