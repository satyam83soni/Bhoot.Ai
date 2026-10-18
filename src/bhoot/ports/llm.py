"""LLM access. Any provider (cloud or on-prem) sits behind ``LLMClient``."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from bhoot.domain.cost import Usage
from bhoot.domain.hashing import content_hash

_FROZEN = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class ContentPart(BaseModel):
    model_config = _FROZEN

    type: Literal["text", "image"]
    text: str | None = None
    image_ref: str | None = None  # blob store key; the adapter loads and encodes it
    detail: Literal["low", "high", "auto"] = "auto"


class LLMMessage(BaseModel):
    model_config = _FROZEN

    role: Literal["system", "user", "assistant"]
    parts: tuple[ContentPart, ...]

    @classmethod
    def text(cls, role: Literal["system", "user", "assistant"], text: str) -> LLMMessage:
        return cls(role=role, parts=(ContentPart(type="text", text=text),))


class ResponseFormat(BaseModel):
    model_config = _FROZEN

    name: str
    json_schema: Mapping[str, Any]  # already adapted to the provider's dialect
    strict: bool = True


class LLMRequest(BaseModel):
    model_config = _FROZEN

    model: str
    messages: tuple[LLMMessage, ...]
    response_format: ResponseFormat | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    seed: int | None = 7
    metadata: Mapping[str, str] = Field(default_factory=dict)  # job, item, tier, node (not hashed)

    def cache_key(self) -> str:
        """Content address of the request. Images are referenced by blob key, which is itself
        content-addressed, so equal images give equal keys. Metadata is excluded."""
        return content_hash(self.model_dump(mode="json", exclude={"metadata"}))


class LLMResponse(BaseModel):
    model_config = _FROZEN

    content: str
    model: str
    parsed: Mapping[str, Any] | None = None
    usage: Usage = Usage()
    cost_usd: Decimal = Decimal("0")
    latency_ms: int = 0
    cache_hit: bool = False
    finish_reason: str | None = None


class ModelCapabilities(BaseModel):
    model_config = _FROZEN

    vision: bool
    json_schema: bool
    strict_schema: bool
    max_input_tokens: int | None = None
    dialect: str = "prompt_only"  # openai_strict | anthropic | gemini | prompt_only | <plugin>


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    def capabilities(self, model: str) -> ModelCapabilities: ...

    def count_tokens(self, model: str, text: str) -> int | None: ...
