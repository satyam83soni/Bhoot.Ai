"""Model tiers: which LLM plays the cheap, strong and cross-check roles.

Each model names its provider. Either set ``provider`` explicitly or write the model as
``provider/model`` (e.g. ``openai/gpt-4.1-mini``). ``openai`` also covers Azure OpenAI and any
OpenAI-compatible server (vLLM, Ollama, LM Studio) via ``api_base``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bhoot.domain.errors import ModelConfigError

Role = Literal["cheap", "strong", "cross"]
Provider = Literal["openai", "anthropic", "gemini"]
PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini")


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str  # "gpt-4.1-mini", or "openai/gpt-4.1-mini" when provider is omitted
    provider: Provider | None = None
    api_base: str | None = None  # Azure / on-prem / OpenAI-compatible endpoints
    api_key_env: str | None = None  # NAME of the env var holding the key, never the key
    vision: bool | None = None  # override the provider default
    json_schema: bool | None = None  # native structured output; None = provider default
    rpm: int | None = Field(default=None, ge=1)
    tpm: int | None = Field(default=None, ge=1)
    input_cost_per_mtok: float | None = Field(default=None, ge=0)  # USD per million tokens
    output_cost_per_mtok: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _provider_known(self) -> ModelSpec:
        if self.provider is None:
            prefix, _, name = self.model.partition("/")
            if prefix not in PROVIDERS or not name:
                raise ValueError(
                    f"model {self.model!r}: set provider ({', '.join(PROVIDERS)}) "
                    "or write it as provider/model"
                )
        return self

    @property
    def provider_name(self) -> Provider:
        if self.provider is not None:
            return self.provider
        return self.model.partition("/")[0]  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """The id sent to the provider (without the provider/ prefix)."""
        prefix, _, name = self.model.partition("/")
        return name if self.provider is None or prefix == self.provider else self.model

    @property
    def model_id(self) -> str:
        """Stable id used inside bhoot: provider/model."""
        return f"{self.provider_name}/{self.model_name}"

    @property
    def has_pricing(self) -> bool:
        return self.input_cost_per_mtok is not None and self.output_cost_per_mtok is not None


class ModelTierConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cheap: ModelSpec
    strong: ModelSpec
    cross: ModelSpec | None = None  # a different model family gives independent errors
    vision_cheap: ModelSpec | None = None  # used when the cheap model can't read images

    def all_specs(self) -> list[ModelSpec]:
        return [
            s for s in (self.cheap, self.strong, self.cross, self.vision_cheap) if s is not None
        ]

    def by_id(self, model_id: str) -> ModelSpec | None:
        return next((s for s in self.all_specs() if s.model_id == model_id), None)

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
            f"the '{role}' model ({spec.model_id}) cannot read images; configure a vision-capable "
            "model (models.vision_cheap for the cheap role)"
        )

    @staticmethod
    def _sees(spec: ModelSpec, has_vision: Callable[[str], bool]) -> bool:
        return spec.vision if spec.vision is not None else has_vision(spec.model_id)
