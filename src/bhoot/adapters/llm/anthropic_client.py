"""Anthropic (Claude) through the official ``anthropic`` SDK.

Structured output uses a forced tool call: the schema is given as the tool's input schema and
the model must call that tool, so the answer is always a JSON object matching it.
"""

from __future__ import annotations

import json
import time
from typing import Any

from bhoot.adapters.llm.common import api_key, cost_from_pricing, image_b64, load_sdk
from bhoot.config.models import ModelSpec
from bhoot.domain.cost import Usage
from bhoot.domain.errors import (
    LLMError,
    ModelCapabilityError,
    ProviderAuthError,
    ProviderTransientError,
    RateLimitedError,
)
from bhoot.ports.llm import LLMMessage, LLMRequest, LLMResponse, ModelCapabilities
from bhoot.ports.storage import BlobStore

DEFAULT_MAX_TOKENS = 8192  # Anthropic requires an explicit limit

_ERRORS = {
    "RateLimitError": RateLimitedError,
    "APITimeoutError": ProviderTransientError,
    "APIConnectionError": ProviderTransientError,
    "InternalServerError": ProviderTransientError,
    "OverloadedError": ProviderTransientError,
    "ServiceUnavailableError": ProviderTransientError,
    "DeadlineExceededError": ProviderTransientError,
    "AuthenticationError": ProviderAuthError,
    "PermissionDeniedError": ProviderAuthError,
    "BadRequestError": ModelCapabilityError,
    "NotFoundError": ModelCapabilityError,
    "RequestTooLargeError": ModelCapabilityError,
    "UnprocessableEntityError": ModelCapabilityError,
}


def map_error(exc: Exception, model: str) -> LLMError:
    return _ERRORS.get(type(exc).__name__, LLMError)(f"{model}: {exc}"[:500])


class AnthropicClient:
    def __init__(
        self,
        spec: ModelSpec,
        blobs: BlobStore | None = None,
        *,
        timeout_s: float = 120.0,
        sdk_client: Any = None,
    ) -> None:
        self.spec = spec
        self.blobs = blobs
        if sdk_client is None:
            anthropic = load_sdk("anthropic")
            sdk_client = anthropic.AsyncAnthropic(
                api_key=api_key(spec, "ANTHROPIC_API_KEY"),
                base_url=spec.api_base,
                timeout=timeout_s,
                max_retries=0,
            )
        self.sdk = sdk_client

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=self.spec.vision if self.spec.vision is not None else True,
            json_schema=self.spec.json_schema if self.spec.json_schema is not None else True,
            strict_schema=False,
            dialect="anthropic",
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        system = "\n\n".join(
            "\n".join(p.text or "" for p in m.parts) for m in request.messages if m.role == "system"
        )
        kwargs: dict[str, Any] = {
            "model": self.spec.model_name,
            "messages": [await self._message(m) for m in request.messages if m.role != "system"],
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": request.temperature,
        }
        if system:
            kwargs["system"] = system
        tool_name = None
        if request.response_format is not None:
            tool_name = request.response_format.name
            kwargs["tools"] = [{
                "name": tool_name,
                "description": "Return the extracted data.",
                "input_schema": dict(request.response_format.json_schema),
            }]  # fmt: skip
            kwargs["tool_choice"] = {"type": "tool", "name": tool_name}
        started = time.perf_counter()
        try:
            response = await self.sdk.messages.create(**kwargs)
        except Exception as exc:
            raise map_error(exc, self.spec.model_id) from exc

        parsed: dict[str, Any] | None = None
        text_parts: list[str] = []
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == tool_name
            ):
                parsed = dict(block.input)
            elif getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        content = json.dumps(parsed) if parsed is not None else "".join(text_parts)
        raw = response.usage
        usage = Usage(
            prompt_tokens=int(getattr(raw, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(raw, "output_tokens", 0) or 0),
            cached_tokens=int(getattr(raw, "cache_read_input_tokens", 0) or 0),
        )
        return LLMResponse(
            content=content,
            model=self.spec.model_id,
            parsed=parsed,
            usage=usage,
            cost_usd=cost_from_pricing(self.spec, usage),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=getattr(response, "stop_reason", None),
        )

    async def _message(self, message: LLMMessage) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for part in message.parts:
            if part.type == "text":
                content.append({"type": "text", "text": part.text or ""})
            else:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": await image_b64(part, self.blobs)},
                })  # fmt: skip
        return {"role": message.role, "content": content}
