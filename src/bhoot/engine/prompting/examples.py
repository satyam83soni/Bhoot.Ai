"""Few-shot example selection (Strategy, with a token cap as a Decorator)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Protocol

from bhoot.domain.document import PageKind
from bhoot.domain.example import Example
from bhoot.engine.registry import StrategyRegistry

CHARS_PER_TOKEN = 4
DEFAULT_MAX_EXAMPLE_TOKENS = 2_500


def example_tokens(example: Example) -> int:
    if example.token_estimate:
        return example.token_estimate
    text = (example.input_text or "") + json.dumps(example.output, ensure_ascii=False, default=str)
    return len(text) // CHARS_PER_TOKEN + (1_600 if example.input_image_ref else 0)


class ExampleSelector(Protocol):
    def select(
        self, examples: Sequence[Example], *, section: str, page_kinds: Iterable[PageKind]
    ) -> tuple[Example, ...]: ...


class StaticSelector:
    """The first ``k`` examples, whatever the pages look like."""

    def __init__(self, k: int = 2) -> None:
        self.k = k

    def select(
        self, examples: Sequence[Example], *, section: str, page_kinds: Iterable[PageKind]
    ) -> tuple[Example, ...]:
        return tuple(examples[: self.k])


class ByPageKindSelector:
    """Prefer examples for the same section and the same kind of page (text vs scanned)."""

    def __init__(self, k: int = 2) -> None:
        self.k = k

    def select(
        self, examples: Sequence[Example], *, section: str, page_kinds: Iterable[PageKind]
    ) -> tuple[Example, ...]:
        kinds = set(page_kinds)

        def score(pair: tuple[int, Example]) -> tuple[int, int, int]:
            index, ex = pair
            same_section = ex.section in (None, section)
            return (0 if same_section else 1, 0 if ex.page_kind in kinds else 1, index)

        usable = [(i, e) for i, e in enumerate(examples) if e.section in (None, section)]
        return tuple(e for _, e in sorted(usable, key=score)[: self.k])


class TokenCappedSelector:
    """Drop examples once their combined size passes ``max_tokens`` (keeps the order)."""

    def __init__(
        self, inner: ExampleSelector, max_tokens: int = DEFAULT_MAX_EXAMPLE_TOKENS
    ) -> None:
        self.inner, self.max_tokens = inner, max_tokens

    def select(
        self, examples: Sequence[Example], *, section: str, page_kinds: Iterable[PageKind]
    ) -> tuple[Example, ...]:
        chosen: list[Example] = []
        used = 0
        for example in self.inner.select(examples, section=section, page_kinds=page_kinds):
            cost = example_tokens(example)
            if used + cost > self.max_tokens:
                break
            chosen.append(example)
            used += cost
        return tuple(chosen)


EXAMPLE_SELECTORS: StrategyRegistry[ExampleSelector] = StrategyRegistry(
    "example selector",
    {
        "static": TokenCappedSelector(StaticSelector()),
        "by_page_kind": TokenCappedSelector(ByPageKindSelector()),
        "none": StaticSelector(k=0),
    },
)
