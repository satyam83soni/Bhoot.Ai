"""OpenAI through the official ``openai`` SDK. Also Azure OpenAI and any OpenAI-compatible
server (vLLM, Ollama, LM Studio) by setting ``api_base``."""

from __future__ import annotations

import time
from typing import Any

from bhoot.adapters.llm.common import api_key, cost_from_pricing, image_b64, load_sdk, safe_json
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

_ERRORS = {
    "RateLimitError": RateLimitedError,
    "APITimeoutError": ProviderTransientError,
    "APIConnectionError": ProviderTransientError,
    "InternalServerError": ProviderTransientError,
    "AuthenticationError": ProviderAuthError,
    "PermissionDeniedError": ProviderAuthError,
    "BadRequestError": ModelCapabilityError,
    "NotFoundError": ModelCapabilityError,
    "UnprocessableEntityError": ModelCapabilityError,
}


def map_error(exc: Exception, model: str) -> LLMError:
    kind = _ERRORS.get(type(exc).__name__, LLMError)
    status = getattr(exc, "status_code", None)
    if kind is LLMError and isinstance(status, int) and status >= 500:
        kind = ProviderTransientError
    return kind(f"{model}: {exc}"[:500])


class OpenAIClient:
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
            openai = load_sdk("openai")
            sdk_client = openai.AsyncOpenAI(
                api_key=api_key(spec, "OPENAI_API_KEY")
                or ("not-needed" if spec.api_base else None),
                base_url=spec.api_base,
                timeout=timeout_s,
                max_retries=0,  # retries are bhoot's job (RetryingLLM)
            )
        self.sdk = sdk_client

    def capabilities(self) -> ModelCapabilities:
        custom_server = self.spec.api_base is not None
        json_schema = (
            self.spec.json_schema if self.spec.json_schema is not None else not custom_server
        )
        return ModelCapabilities(
            vision=self.spec.vision if self.spec.vision is not None else not custom_server,
            json_schema=json_schema,
            strict_schema=json_schema,
            dialect="openai_strict" if json_schema else "prompt_only",
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.spec.model_name,
            "messages": [await self._message(m) for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.response_format is not None:
            if self.capabilities().json_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.response_format.name,
                        "schema": dict(request.response_format.json_schema),
                        "strict": request.response_format.strict,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            response = await self.sdk.chat.completions.create(**kwargs)
        except Exception as exc:
            raise map_error(exc, self.spec.model_id) from exc
        choice = response.choices[0]
        content = choice.message.content or ""
        raw_usage = getattr(response, "usage", None)
        details = getattr(raw_usage, "prompt_tokens_details", None)
        usage = Usage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            cached_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        )
        return LLMResponse(
            content=content,
            model=self.spec.model_id,
            parsed=safe_json(content),
            usage=usage,
            cost_usd=cost_from_pricing(self.spec, usage),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=getattr(choice, "finish_reason", None),
        )

    async def _message(self, message: LLMMessage) -> dict[str, Any]:
        if all(p.type == "text" for p in message.parts):
            return {"role": message.role, "content": "\n".join(p.text or "" for p in message.parts)}
        parts: list[dict[str, Any]] = []
        for part in message.parts:
            if part.type == "text":
                parts.append({"type": "text", "text": part.text or ""})
            else:
                url = "data:image/png;base64," + await image_b64(part, self.blobs)
                parts.append(
                    {"type": "image_url", "image_url": {"url": url, "detail": part.detail}}
                )
        return {"role": message.role, "content": parts}
