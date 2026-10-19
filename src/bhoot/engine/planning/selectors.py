"""Page selectors: which pages a section looks at (Strategy + Composite).

Selecting fewer pages is the cheapest way to save tokens: a summary that sits on the first
page doesn't need the other forty sent along with it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from bhoot.domain.document import Document, PageKind
from bhoot.domain.errors import ConfigError
from bhoot.domain.pack import PageSelectorSpec
from bhoot.engine.registry import StrategyRegistry


class PageSelector(Protocol):
    def select(self, doc: Document) -> tuple[int, ...]: ...  # 0-based, ascending, unique


SelectorFactory = Callable[[PageSelectorSpec, "SelectorBuilder"], PageSelector]


class AllPages:
    def select(self, doc: Document) -> tuple[int, ...]:
        return tuple(range(doc.page_count))


class FirstN:
    def __init__(self, n: int) -> None:
        self.n = n

    def select(self, doc: Document) -> tuple[int, ...]:
        return tuple(range(min(self.n, doc.page_count)))


class LastN:
    def __init__(self, n: int) -> None:
        self.n = n

    def select(self, doc: Document) -> tuple[int, ...]:
        return tuple(range(max(0, doc.page_count - self.n), doc.page_count))


class KeywordPages:
    """Pages whose text contains any keyword (case-insensitive). Pages without readable text
    are kept, because we can't rule them out."""

    def __init__(self, keywords: tuple[str, ...]) -> None:
        self.keywords = tuple(k.casefold() for k in keywords)

    def select(self, doc: Document) -> tuple[int, ...]:
        out = []
        for page in doc.pages:
            text = page.text.casefold()
            if not text.strip() or any(k in text for k in self.keywords):
                out.append(page.index)
        return tuple(out)


class ByPageKind:
    def __init__(self, kinds: tuple[PageKind, ...]) -> None:
        self.kinds = frozenset(kinds)

    def select(self, doc: Document) -> tuple[int, ...]:
        return tuple(p.index for p in doc.pages if p.kind in self.kinds)


class UnionSelector:
    def __init__(self, children: tuple[PageSelector, ...]) -> None:
        self.children = children

    def select(self, doc: Document) -> tuple[int, ...]:
        return tuple(sorted({i for child in self.children for i in child.select(doc)}))


def _need_n(spec: PageSelectorSpec) -> int:
    if spec.n is None:
        raise ConfigError(f"page selector {spec.kind!r} needs n")
    return spec.n


def _need(value: tuple[object, ...], what: str, kind: str) -> None:
    if not value:
        raise ConfigError(f"page selector {kind!r} needs {what}")


def _keyword(spec: PageSelectorSpec, build: SelectorBuilder) -> PageSelector:
    _need(spec.keywords, "keywords", spec.kind)
    return KeywordPages(spec.keywords)


def _page_kind(spec: PageSelectorSpec, build: SelectorBuilder) -> PageSelector:
    _need(spec.page_kinds, "page_kinds", spec.kind)
    return ByPageKind(spec.page_kinds)


def _union(spec: PageSelectorSpec, build: SelectorBuilder) -> PageSelector:
    _need(spec.children, "children", spec.kind)
    return UnionSelector(tuple(build.build(child) for child in spec.children))


SELECTORS: StrategyRegistry[SelectorFactory] = StrategyRegistry(
    "page selector",
    {
        "all": lambda spec, build: AllPages(),
        "first_n": lambda spec, build: FirstN(_need_n(spec)),
        "last_n": lambda spec, build: LastN(_need_n(spec)),
        "keyword": _keyword,
        "page_kind": _page_kind,
        "union": _union,
    },
)


class SelectorBuilder:
    def __init__(self, registry: StrategyRegistry[SelectorFactory] = SELECTORS) -> None:
        self.registry = registry

    def build(self, spec: PageSelectorSpec) -> PageSelector:
        return self.registry.get(spec.kind)(spec, self)
