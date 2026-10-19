"""Routes each request to the right provider client, chosen from config (``provider:``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from bhoot.adapters.llm.anthropic_client import AnthropicClient
from bhoot.adapters.llm.gemini_client import GeminiClient
from bhoot.adapters.llm.openai_client import OpenAIClient
from bhoot.config.models import ModelSpec, ModelTierConfig
from bhoot.domain.errors import ModelConfigError
from bhoot.ports.llm import LLMRequest, LLMResponse, ModelCapabilities
from bhoot.ports.storage import BlobStore


class ProviderClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    def capabilities(self) -> ModelCapabilities: ...


ClientFactory = Callable[[ModelSpec, BlobStore | None, float], ProviderClient]

DEFAULT_FACTORIES: dict[str, ClientFactory] = {
    "openai": lambda spec, blobs, timeout: OpenAIClient(spec, blobs, timeout_s=timeout),
    "anthropic": lambda spec, blobs, timeout: AnthropicClient(spec, blobs, timeout_s=timeout),
    "gemini": lambda spec, blobs, timeout: GeminiClient(spec, blobs, timeout_s=timeout),
}


class ModelRouter:
    """An LLMClient over all configured models. ``request.model`` is the model id
    (``provider/model``); provider SDK clients are created lazily on first use."""

    def __init__(
        self,
        models: ModelTierConfig,
        blobs: BlobStore | None = None,
        *,
        timeout_s: float = 120.0,
        factories: dict[str, ClientFactory] | None = None,
    ) -> None:
        self.models = models
        self.blobs = blobs
        self.timeout_s = timeout_s
        self.factories = factories or DEFAULT_FACTORIES
        self._clients: dict[str, ProviderClient] = {}

    def client_for(self, model_id: str) -> ProviderClient:
        if model_id not in self._clients:
            spec = self.models.by_id(model_id)
            if spec is None:
                known = ", ".join(s.model_id for s in self.models.all_specs())
                raise ModelConfigError(
                    f"model {model_id!r} is not configured (configured: {known})"
                )
            self._clients[model_id] = self.factories[spec.provider_name](
                spec, self.blobs, self.timeout_s
            )
        return self._clients[model_id]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self.client_for(request.model).complete(request)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.client_for(model).capabilities()

    def count_tokens(self, model: str, text: str) -> int | None:
        return None  # callers estimate (chars / 4); exact counts aren't needed for budgeting
