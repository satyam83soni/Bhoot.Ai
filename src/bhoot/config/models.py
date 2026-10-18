"""Model tiers: which LLM plays the cheap, strong and cross-check roles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from bhoot.domain.errors import ModelConfigError

Role = Literal["cheap", "strong", "cross"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str  # any LiteLLM id: "openai/…", "anthropic/…", "gemini/…", "hosted_vllm/…"
    api_base: str | None = None  # on-prem / OpenAI-compatible endpoints
    api_key_env: str | None = None  # NAME of the env var holding the key, never the key
    vision: bool | None = None  # override capability detection
    json_schema: bool | None = None
    rpm: int | None = Field(default=None, ge=1)
    tpm: int | None = Field(default=None, ge=1)
    input_cost_per_mtok: float | None = Field(default=None, ge=0)  # e.g. 0 for on-prem
    output_cost_per_mtok: float | None = Field(default=None, ge=0)


class ModelTierConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cheap: ModelSpec
    strong: ModelSpec
    cross: ModelSpec | None = None  # a different model family gives independent errors
    vision_cheap: ModelSpec | None = None  # used when the cheap model can't read images

    def resolve(
        self, role: Role, *, needs_vision: bool, has_vision: Callable[[str], bool]
    ) -> ModelSpec:
        """The model to call for a role. Swaps in ``vision_cheap`` when the cheap model can't
        see images; raises if no configured model can."""
        spec = {"cheap": self.cheap, "strong": self.strong, "cross": self.cross}[role]
        if spec is None:
            raise ModelConfigError(f"no '{role}' model is configured")
        if not needs_vision or self._sees(spec, has_vision):
            return spec
        if role == "cheap" and self.vision_cheap and self._sees(self.vision_cheap, has_vision):
            return self.vision_cheap
        raise ModelConfigError(
            f"the '{role}' model ({spec.model}) cannot read images; configure a vision-capable "
            "model (models.vision_cheap for the cheap role)"
        )

    @staticmethod
    def _sees(spec: ModelSpec, has_vision: Callable[[str], bool]) -> bool:
        return spec.vision if spec.vision is not None else has_vision(spec.model)
