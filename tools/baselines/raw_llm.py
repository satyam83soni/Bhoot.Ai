"""Raw LLM baseline: one strict-JSON call per document, no verification or repair.

This is what most teams do today. Any LiteLLM model id works (``openai/…``, ``anthropic/…``,
``gemini/…``, ``hosted_vllm/…``). The completion function is injectable so tests run offline.
"""

from __future__ import annotations

import json
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


def _default_completion(**kwargs: Any) -> Any:
    import litellm  # imported lazily: only needed for live runs

    return litellm.completion(**kwargs)


def _cost(response: Any) -> float:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception:  # pricing unknown for custom / on-prem models
        return 0.0


class RawLLMBaseline(BaselineRunner):
    def __init__(
        self, model: str, *, completion: Completion | None = None, max_tokens: int = 16_000
    ) -> None:
        self.model = model
        self.name = f"raw_llm:{model}"
        self._completion = completion or _default_completion
        self._max_tokens = max_tokens

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
                model=self.model,
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
            return Prediction(json.loads(text), cost_usd=_cost(response))
        except Exception as exc:  # a failed call is a result, not a crash: it scores as all-missed
            return Prediction(None, error=f"{type(exc).__name__}: {exc}"[:300])
