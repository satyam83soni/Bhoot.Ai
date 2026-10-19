"""Provider dialects: the same envelope, shaped for what each structured-output API accepts.

* ``openai_strict``: JSON Schema with ``strict: true`` (the envelope is already strict-ready);
* ``anthropic``: the schema becomes a forced tool's ``input_schema``;
* ``gemini``: ``response_json_schema`` with nullability written as ``anyOf`` and no
  ``additionalProperties``;
* ``prompt_only``: no native support (custom servers): the schema goes into the prompt and the
  answer is parsed leniently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Protocol

from bhoot.engine.registry import StrategyRegistry
from bhoot.ports.llm import ModelCapabilities, ResponseFormat

_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def format_name(name: str) -> str:
    return _NAME.sub("_", name)[:64] or "extraction"


class SchemaDialect(Protocol):
    name: str

    def adapt(self, envelope: Mapping[str, Any], name: str) -> ResponseFormat | None: ...

    def instructions(self, envelope: Mapping[str, Any]) -> str: ...


class OpenAIStrictDialect:
    name = "openai_strict"

    def adapt(self, envelope: Mapping[str, Any], name: str) -> ResponseFormat | None:
        return ResponseFormat(name=format_name(name), json_schema=dict(envelope), strict=True)

    def instructions(self, envelope: Mapping[str, Any]) -> str:
        return ""


class AnthropicDialect:
    name = "anthropic"
    _UNSUPPORTED = frozenset({"$schema", "$id", "examples"})

    def adapt(self, envelope: Mapping[str, Any], name: str) -> ResponseFormat | None:
        schema = _strip_keys(envelope, self._UNSUPPORTED)
        return ResponseFormat(name=format_name(name), json_schema=schema, strict=False)

    def instructions(self, envelope: Mapping[str, Any]) -> str:
        return "Give your answer by calling the tool; put every field in its arguments."


class GeminiDialect:
    name = "gemini"

    def adapt(self, envelope: Mapping[str, Any], name: str) -> ResponseFormat | None:
        return ResponseFormat(name=format_name(name), json_schema=_gemini(envelope), strict=False)

    def instructions(self, envelope: Mapping[str, Any]) -> str:
        return ""


class PromptOnlyDialect:
    name = "prompt_only"

    def adapt(self, envelope: Mapping[str, Any], name: str) -> ResponseFormat | None:
        return None

    def instructions(self, envelope: Mapping[str, Any]) -> str:
        schema = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        return (
            "Reply with exactly one JSON object and nothing else (no prose, no code fences). "
            f"It must match this JSON Schema:\n{schema}"
        )


DIALECTS: StrategyRegistry[SchemaDialect] = StrategyRegistry(
    "schema dialect",
    {
        "openai_strict": OpenAIStrictDialect(),
        "anthropic": AnthropicDialect(),
        "gemini": GeminiDialect(),
        "prompt_only": PromptOnlyDialect(),
    },
)


def dialect_for(
    caps: ModelCapabilities, registry: StrategyRegistry[SchemaDialect] = DIALECTS
) -> tuple[SchemaDialect, str | None]:
    """The dialect for a model, and a warning when we had to fall back to prompt-only."""
    if caps.dialect in registry:
        return registry.get(caps.dialect), None
    warning = f"unknown schema dialect {caps.dialect!r}: the schema is sent in the prompt instead"
    return registry.get("prompt_only"), warning


def _strip_keys(node: Any, drop: frozenset[str]) -> Any:
    if isinstance(node, Mapping):
        return {k: _strip_keys(v, drop) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_strip_keys(v, drop) for v in node]
    return node


def _gemini(node: Any) -> Any:
    if isinstance(node, list):
        return [_gemini(v) for v in node]
    if not isinstance(node, Mapping):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "additionalProperties":
            continue
        out[key] = _gemini(value)
    kind = out.get("type")
    if isinstance(kind, list) and "null" in kind:
        rest = [t for t in kind if t != "null"]
        del out["type"]
        base = {"type": rest[0] if len(rest) == 1 else rest}
        out["anyOf"] = [base, {"type": "null"}]
    return out
