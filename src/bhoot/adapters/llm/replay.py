"""Record real LLM answers once, replay them forever (cassettes for tests and the CI eval gate).

Cassettes are JSON files keyed by the request's content hash, so the same request always gets
the same recorded answer and CI never needs API keys or spends money.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from bhoot.domain.errors import LLMError
from bhoot.ports.llm import LLMClient, LLMRequest, LLMResponse, ModelCapabilities

Mode = Literal["replay", "record", "auto"]


class CassetteMissError(LLMError):
    """Replay mode: no recorded answer for this request."""


class ReplayLLMClient:
    def __init__(
        self, cassette: Path, *, inner: LLMClient | None = None, mode: Mode = "replay"
    ) -> None:
        if mode != "replay" and inner is None:
            raise ValueError("record/auto mode needs a real client to call")
        self.cassette = cassette
        self.inner = inner
        self.mode = mode
        self._entries: dict[str, dict[str, object]] = (
            json.loads(cassette.read_text(encoding="utf-8")) if cassette.exists() else {}
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key()
        if key in self._entries and self.mode != "record":
            return LLMResponse.model_validate(self._entries[key]).model_copy(
                update={"cache_hit": True}
            )
        if self.mode == "replay" or self.inner is None:
            raise CassetteMissError(
                f"no recorded answer in {self.cassette.name} for {request.model} "
                f"({dict(request.metadata)}); re-record with mode='record'"
            )
        response = await self.inner.complete(request)
        self._entries[key] = response.model_dump(mode="json")
        self._save()
        return response

    def _save(self) -> None:
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._entries, indent=1, sort_keys=True) + "\n"
        self.cassette.write_bytes(text.encode("utf-8"))

    def capabilities(self, model: str) -> ModelCapabilities:
        if self.inner is not None:
            return self.inner.capabilities(model)
        return ModelCapabilities(
            vision=True, json_schema=True, strict_schema=True, dialect="openai_strict"
        )

    def count_tokens(self, model: str, text: str) -> int | None:
        return self.inner.count_tokens(model, text) if self.inner else len(text) // 4
