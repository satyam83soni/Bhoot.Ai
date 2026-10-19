"""Chunkers: how a section's pages are grouped into LLM calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from bhoot.domain.document import Document, PageKind
from bhoot.domain.errors import ConfigError
from bhoot.domain.pack import ChunkSpec
from bhoot.engine.cost.pricing import CHARS_PER_TOKEN, DEFAULT_IMAGE_TOKENS
from bhoot.engine.registry import StrategyRegistry


class Chunker(Protocol):
    def split(self, pages: Sequence[int], doc: Document) -> list[tuple[int, ...]]: ...


def page_tokens(doc: Document, index: int) -> int:
    page = doc.page(index)
    if page.kind in (PageKind.SCANNED, PageKind.MIXED) or not page.text.strip():
        return DEFAULT_IMAGE_TOKENS
    return len(page.text) // CHARS_PER_TOKEN + 1


class PerPageChunker:
    """One call per page: best for rows that don't depend on other pages."""

    def split(self, pages: Sequence[int], doc: Document) -> list[tuple[int, ...]]:
        return [(p,) for p in pages]


class TokenBudgetChunker:
    """Neighbouring pages together until the next page would pass ``max_tokens``. A page that
    alone is bigger than the budget still gets its own call."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def split(self, pages: Sequence[int], doc: Document) -> list[tuple[int, ...]]:
        chunks: list[tuple[int, ...]] = []
        current: list[int] = []
        used = 0
        for page in pages:
            cost = page_tokens(doc, page)
            adjacent = not current or page == current[-1] + 1
            if current and (used + cost > self.max_tokens or not adjacent):
                chunks.append(tuple(current))
                current, used = [], 0
            current.append(page)
            used += cost
        if current:
            chunks.append(tuple(current))
        return chunks


class WindowChunker:
    """Windows of ``size`` pages overlapping by ``overlap`` pages, for content that runs
    across a page break."""

    def __init__(self, size: int, overlap: int) -> None:
        if size < 1 or not 0 <= overlap < size:
            raise ConfigError(
                f"window chunker needs size >= 1 and 0 <= overlap < size, got {size}/{overlap}"
            )
        self.size, self.overlap = size, overlap

    def split(self, pages: Sequence[int], doc: Document) -> list[tuple[int, ...]]:
        pages = list(pages)
        if len(pages) <= self.size:
            return [tuple(pages)] if pages else []
        step = self.size - self.overlap
        chunks = []
        for start in range(0, len(pages), step):
            chunks.append(tuple(pages[start : start + self.size]))
            if start + self.size >= len(pages):
                break
        return chunks


ChunkerFactory = Callable[[ChunkSpec], Chunker]

CHUNKERS: StrategyRegistry[ChunkerFactory] = StrategyRegistry(
    "chunker",
    {
        "per_page": lambda spec: PerPageChunker(),
        "token_budget": lambda spec: TokenBudgetChunker(spec.max_tokens),
        "window": lambda spec: WindowChunker(int(spec.params.get("size", 2)), spec.overlap_pages),
    },
)


def build_chunker(
    spec: ChunkSpec, registry: StrategyRegistry[ChunkerFactory] = CHUNKERS
) -> Chunker:
    return registry.get(spec.kind)(spec)
