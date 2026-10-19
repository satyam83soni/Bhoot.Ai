"""Document packs: schema + prompt + examples + sections + rules + optional Tier-0 extractor.

Packs never touch ports or adapters. Everything a code extractor needs (pages, tables, value
parsers, options) arrives through ``CodeExtractionContext``; the engine decides how tables are
produced (ruled PDF tables, words from borderless tables, or OCR words).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from bhoot.domain.candidate import CandidateValue
from bhoot.domain.document import CropRequest, Document, PageKind, Table
from bhoot.domain.example import Example
from bhoot.domain.locales import Locale
from bhoot.domain.prompt import PromptTemplate
from bhoot.domain.result import FieldResult
from bhoot.domain.rules import Rule
from bhoot.domain.schema import SchemaDefinition
from bhoot.domain.values import ParsedAmount

__all__ = [
    "ChunkSpec",
    "CodeExtraction",
    "CodeExtractionContext",
    "CodeExtractor",
    "CropRequest",
    "DocPack",
    "Locale",
    "PageSelectorSpec",
    "ResultPostProcessor",
    "SectionSpec",
    "TableProvider",
    "ValueParsers",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class PageSelectorSpec(BaseModel):
    """Where a section looks. ``kind`` is a name in the engine's selector registry."""

    model_config = _FROZEN

    kind: str = "all"  # all | first_n | last_n | keyword | page_kind | union | <plugin>
    n: int | None = Field(default=None, ge=1)
    keywords: tuple[str, ...] = ()
    page_kinds: tuple[PageKind, ...] = ()
    children: tuple[PageSelectorSpec, ...] = ()
    params: Mapping[str, Any] = Field(default_factory=dict)


class ChunkSpec(BaseModel):
    """How selected pages are grouped into LLM calls. ``kind`` is a chunker registry name."""

    model_config = _FROZEN

    kind: str = "per_page"  # per_page | token_budget | window | <plugin>
    max_tokens: int = Field(default=6000, ge=100)
    overlap_pages: int = Field(default=0, ge=0)
    params: Mapping[str, Any] = Field(default_factory=dict)


class SectionSpec(BaseModel):
    """A slice of the schema plus where to find it (page pruning)."""

    model_config = _FROZEN

    name: str
    paths: tuple[str, ...]  # schema field patterns produced by this section; "*" = everything
    pages: PageSelectorSpec = Field(default_factory=PageSelectorSpec)
    chunking: ChunkSpec = Field(default_factory=ChunkSpec)
    merge: str = "scalar_prefer_verified"  # merge strategy registry name


# ---------------------------------------------------------------- Tier-0 contract
class TableProvider(Protocol):
    def tables(
        self, page_index: int, *, header_matcher: Callable[[Sequence[str]], bool] | None = None
    ) -> tuple[Table, ...]: ...


@dataclass(frozen=True)
class ValueParsers:
    """Amount and date parsers bound to the document's locale (implemented by the engine)."""

    amount: Callable[[str], ParsedAmount]
    date: Callable[[str], date | None]


@dataclass(frozen=True)
class CodeExtractionContext:
    doc: Document
    section: SectionSpec
    locale: Locale
    tables: TableProvider
    values: ValueParsers
    pack_options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class CodeExtraction:
    candidates: tuple[CandidateValue, ...] = ()
    covered_paths: frozenset[str] = frozenset()
    covered_sections: frozenset[tuple[str, int]] = frozenset()  # (section, page) needing no LLM
    crop_requests: tuple[CropRequest, ...] = ()
    notes: tuple[str, ...] = ()


@runtime_checkable
class CodeExtractor(Protocol):
    """Tier 0: returns only values it can check itself; the LLM fills the rest."""

    def extract(self, ctx: CodeExtractionContext) -> CodeExtraction: ...


@runtime_checkable
class ResultPostProcessor(Protocol):
    """Runs on the final result before caching and output (e.g. Aadhaar masking).
    Must be pure and idempotent."""

    name: str

    def process(
        self,
        data: Mapping[str, Any],
        fields: Mapping[str, FieldResult],
        pack_options: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, FieldResult]]: ...


# ---------------------------------------------------------------- the pack
@dataclass(frozen=True)
class DocPack:
    name: str  # e.g. "lending.bank_statement"
    version: int
    schema: SchemaDefinition
    prompt: PromptTemplate
    sections: tuple[SectionSpec, ...]
    description: str = ""
    examples: tuple[Example, ...] = ()
    rules: tuple[Rule, ...] = ()
    code_extractor: CodeExtractor | None = None
    locale: Locale = field(default_factory=Locale)
    postprocessors: tuple[ResultPostProcessor, ...] = ()
    default_options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        names = [s.name for s in self.sections]
        if not names:
            raise ValueError(f"pack {self.name}: at least one section is required")
        if len(set(names)) != len(names):
            raise ValueError(f"pack {self.name}: section names must be unique")
        ids = [r.rule_id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"pack {self.name}: rule ids must be unique")

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def section(self, name: str) -> SectionSpec:
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(f"pack {self.name} has no section {name!r}")
