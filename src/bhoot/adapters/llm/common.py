"""Helpers shared by the provider clients: SDK loading, keys, images, cost, JSON."""

from __future__ import annotations

import base64
import importlib
import json
import os
from decimal import Decimal
from types import ModuleType
from typing import Any

from bhoot.config.models import ModelSpec
from bhoot.domain.cost import Usage
from bhoot.domain.errors import ModelConfigError, ProviderAuthError
from bhoot.ports.llm import ContentPart
from bhoot.ports.storage import BlobStore

EXTRA_FOR = {"openai": "openai", "anthropic": "anthropic", "google.genai": "gemini"}


def load_sdk(module: str) -> ModuleType:
    """Import a provider SDK, or explain which bhoot extra installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        extra = EXTRA_FOR.get(module, module)
        raise ModelConfigError(
            f"the {extra} provider needs its SDK: pip install 'bhoot[{extra}]'"
        ) from exc


def api_key(spec: ModelSpec, default_env: str | None) -> str | None:
    """Key from the env var named in config (or the provider's usual variable)."""
    env = spec.api_key_env or default_env
    if env is None:
        return None
    value = os.environ.get(env)
    if not value and spec.api_key_env:
        raise ProviderAuthError(f"{spec.model_id}: env var {spec.api_key_env} is not set")
    return value or None


async def image_b64(part: ContentPart, blobs: BlobStore | None) -> str:
    if part.image_ref is None or blobs is None:
        raise ModelConfigError("image part without an image_ref or blob store")
    return base64.b64encode(await blobs.get(part.image_ref)).decode("ascii")


def cost_from_pricing(spec: ModelSpec | None, usage: Usage) -> Decimal:
    """USD from the per-model prices in config. No price configured means no cost (budgets
    then can't be enforced; `bhoot doctor` warns about it)."""
    if spec is None:
        return Decimal("0")
    price_in, price_out = spec.input_cost_per_mtok, spec.output_cost_per_mtok
    if price_in is None or price_out is None:
        return Decimal("0")
    usd = (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1_000_000
    return Decimal(str(round(usd, 8)))


def safe_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None
