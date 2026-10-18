"""Prompt templates (pure data; template files are read elsewhere)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bhoot.domain.hashing import content_hash


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str
    version: int
    body: str  # Jinja2 source
    variables: frozenset[str]
    description: str | None
    content_hash: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


def make_prompt_template(
    *, name: str, version: int, body: str, variables: Iterable[str], description: str | None = None
) -> PromptTemplate:
    """Builds a template and its hash. The hash covers what changes behaviour (body and
    variables), not the name, so renaming a prompt does not invalidate results."""
    names = frozenset(variables)
    digest = content_hash({"body": body, "variables": sorted(names)})
    return PromptTemplate(name, version, body, names, description, digest)
