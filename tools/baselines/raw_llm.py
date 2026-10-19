"""Raw LLM baseline: one strict-JSON call per document, no verification or repair.

This is what most teams do today. Models are named ``provider/model`` (``openai/...``,
``anthropic/...``, ``gemini/...``); all three are reached through the OpenAI SDK and the
providers' OpenAI-compatible endpoints, or any other server via ``--api-base``. The completion
function is injectable so tests run offline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from tools.baselines.base import BaselineRunner, Prediction
from tools.baselines.documents import DocumentInput
from tools.baselines.schemas import INSTRUCTIONS, SCHEMAS

Completion = Callable[..., Any]

SYSTEM = (
    "You extract structured data from documents. Return only JSON matching the schema. "
    "Use null when a value is not present. Dates as YYYY-MM-DD, amounts as plain decimals."
)

# provider -> (OpenAI-compatible base URL, env var holding the key)
ENDPOINTS = {
    "openai": (None, "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}


def openai_completion(provider: str, api_base: str | None = None) -> Completion:
    """A completion function backed by the OpenAI SDK, created on first use."""
    client: Any = None

    def complete(**kwargs: Any) -> Any:
        nonlocal client
        if client is None:
            from openai import OpenAI

            base, key_env = ENDPOINTS.get(provider, (None, "OPENAI_API_KEY"))
            key = os.environ.get(key_env) or "not-needed"
            client = OpenAI(base_url=api_base or base, api_key=key)
        return client.chat.completions.create(**kwargs)

    return complete


def split_model(model: str) -> tuple[str, str]:
    provider, _, name = model.partition("/")
    return (provider, name) if name else ("openai", model)


class RawLLMBaseline(BaselineRunner):
    def __init__(
        self,
        model: str,
        *,
        completion: Completion | None = None,
        api_base: str | None = None,
        price_per_mtok: tuple[float, float] = (0.0, 0.0),
        max_tokens: int = 16_000,
    ) -> None:
        provider, self.model_name = split_model(model)
        self.name = f"raw_llm:{model}"
        self._completion = completion or openai_completion(provider, api_base)
        self._price_in, self._price_out = price_per_mtok
        self._max_tokens = max_tokens

    def _cost(self, response: Any) -> float:
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        return float(tokens_in * self._price_in + tokens_out * self._price_out) / 1_000_000

    def messages(self, document: DocumentInput, schema_ref: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": INSTRUCTIONS[schema_ref]}]
        for number, text in enumerate(document.pages_text, start=1):
            if text:
                content.append({"type": "text", "text": f'<page n="{number}">\n{text}\n</page>'})
        for png in document.pages_png_b64:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}
            )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

    def predict(self, document: DocumentInput, schema_ref: str) -> Prediction:
        try:
            response = self._completion(
                model=self.model_name,
                messages=self.messages(document, schema_ref),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": SCHEMAS[schema_ref],
                        "strict": True,
                    },
                },
                temperature=0,
                max_tokens=self._max_tokens,
            )
            text = response.choices[0].message.content or ""
            return Prediction(json.loads(text), cost_usd=self._cost(response))
        except Exception as exc:  # a failed call is a result, not a crash: it scores as all-missed
            return Prediction(None, error=f"{type(exc).__name__}: {exc}"[:300])
