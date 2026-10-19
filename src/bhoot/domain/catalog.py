"""Listing types for registry and pack catalogues."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["schema", "prompt"]
    name: str
    versions: tuple[int, ...]
    source: Literal["project", "pack"]
    hashes: Mapping[int, str] = Field(default_factory=dict)
    migrations: Mapping[int, str] = Field(default_factory=dict)  # version -> migration id


class PackInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: int
    description: str = ""
    distribution: str | None = None  # pip package that registered it
    sections: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
