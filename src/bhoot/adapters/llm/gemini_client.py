"""Google Gemini through the official ``google-genai`` SDK (native JSON schema output)."""

from __future__ import annotations

import base64
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
from bhoot.ports.llm import LLMRequest, LLMResponse, ModelCapabilities
from bhoot.ports.storage import BlobStore


def map_error(exc: Exception, model: str) -> LLMError:
    code = getattr(exc, "code", None)
    text = f"{model}: {exc}"[:500]
    if code == 429:
        return RateLimitedError(text)
    if code in (401, 403):
        return ProviderAuthError(text)
    if isinstance(code, int) and code >= 500:
        return ProviderTransientError(text)
    if isinstance(code, int) and 400 <= code < 500:
        return ModelCapabilityError(text)
    if type(exc).__name__ in ("TimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout"):
        return ProviderTransientError(text)
    return LLMError(text)


class GeminiClient:
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
        self.timeout_s = timeout_s
        if sdk_client is None:
            genai = load_sdk("google.genai")
            sdk_client = genai.Client(api_key=api_key(spec, "GEMINI_API_KEY"))
        self.sdk = sdk_client

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=self.spec.vision if self.spec.vision is not None else True,
            json_schema=self.spec.json_schema if self.spec.json_schema is not None else True,
            strict_schema=False,
            dialect="gemini",
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        system = "\n\n".join(
            "\n".join(p.text or "" for p in m.parts) for m in request.messages if m.role == "system"
        )
        contents = []
        for message in request.messages:
            if message.role == "system":
                continue
            parts: list[dict[str, Any]] = []
            for part in message.parts:
                if part.type == "text":
                    parts.append({"text": part.text or ""})
                else:
                    data = base64.b64decode(await image_b64(part, self.blobs))
                    parts.append({"inline_data": {"mime_type": "image/png", "data": data}})
            contents.append(
                {"role": "model" if message.role == "assistant" else "user", "parts": parts}
            )

        config: dict[str, Any] = {"temperature": request.temperature}
        if system:
            config["system_instruction"] = system
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.seed is not None:
            config["seed"] = request.seed
        if request.response_format is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = dict(request.response_format.json_schema)

        started = time.perf_counter()
        try:
            response = await self.sdk.aio.models.generate_content(
                model=self.spec.model_name, contents=contents, config=config
            )
        except Exception as exc:
            raise map_error(exc, self.spec.model_id) from exc
        content = getattr(response, "text", None) or ""
        meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
            cached_tokens=int(getattr(meta, "cached_content_token_count", 0) or 0),
        )
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        return LLMResponse(
            content=content,
            model=self.spec.model_id,
            parsed=safe_json(content),
            usage=usage,
            cost_usd=cost_from_pricing(self.spec, usage),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=str(finish) if finish is not None else None,
        )
