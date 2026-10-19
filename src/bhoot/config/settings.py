"""All runtime settings. Values come from (highest first): code, BHOOT_* env vars, bhoot.yaml,
built-in defaults. Every key has a default, so bhoot runs without any config file (models
excepted: they must be configured before the first LLM call).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from bhoot.config.models import ModelTierConfig
from bhoot.config.modes import profile_for
from bhoot.domain.modes import Mode, ModeProfile

_SECTION = ConfigDict(extra="forbid")


class VisionSettings(BaseModel):
    model_config = _SECTION
    dpi: int = Field(default=200, ge=50, le=600)
    max_side_px: int = Field(default=2000, ge=256)
    grayscale: bool = True


class OcrSettings(BaseModel):
    model_config = _SECTION
    enabled: bool = True  # used only when bhoot[ocr] is installed
    engine: str = "rapidocr"
    model_profile: Literal["mobile", "server"] = "mobile"
    lang: tuple[str, ...] = ("en",)
    dpi: int = Field(default=300, ge=72, le=600)
    preprocess: Literal["off", "auto", "aggressive"] = "auto"
    min_page_confidence: float = Field(default=0.80, ge=0, le=1)
    min_words: int = Field(default=15, ge=0)
    critical_word_confidence: float = Field(default=0.90, ge=0, le=1)
    max_parallel_pages: int | None = Field(default=None, ge=1)
    threads_per_page: int = Field(default=2, ge=1)

    @field_validator("preprocess", mode="before")
    @classmethod
    def _yaml_off(cls, value: object) -> object:
        # YAML reads a bare `off` / `on` as a boolean; accept it the way users mean it
        if value is False:
            return "off"
        if value is True:
            return "auto"
        return value


class CacheSettings(BaseModel):
    model_config = _SECTION
    results: bool = True
    llm: bool = True
    llm_max_mb: int = Field(default=512, ge=1)
    result_ttl_days: int | None = Field(default=None, ge=1)
    keep_sources: bool = False


class LimitsSettings(BaseModel):
    model_config = _SECTION
    max_file_mb: int = Field(default=50, ge=1)
    max_pages: int = Field(default=500, ge=1)
    max_image_pixels: int = Field(default=89_478_485, ge=1_000_000)
    max_render_side_px: int = Field(default=6000, ge=512)


class RetrySettings(BaseModel):
    model_config = _SECTION
    max_attempts: int = Field(default=5, ge=1)
    initial_s: float = Field(default=1.0, gt=0)
    max_s: float = Field(default=30.0, gt=0)


class RateLimit(BaseModel):
    model_config = _SECTION
    rpm: int | None = Field(default=None, ge=1)
    tpm: int | None = Field(default=None, ge=1)
    max_concurrent: int | None = Field(default=None, ge=1)


class ConcurrencySettings(BaseModel):
    model_config = _SECTION
    max_parallel_llm_calls: int = Field(default=4, ge=1)
    max_parallel_docs: int = Field(default=4, ge=1)


class EngineSettings(BaseModel):
    model_config = _SECTION
    graph: str = "default"
    policy_rules: tuple[str, ...] | None = None
    example_selector: str = "by_page_kind"


class RegistrySettings(BaseModel):
    model_config = _SECTION
    definitions: str | None = None  # module path; default from [tool.bhoot] in pyproject.toml
    migrations_dir: Path = Path("bhoot/migrations")
    strict: bool = False
    allow_draft: bool = True
    cache_max_mb: int = Field(default=64, ge=1)


class TrainingSettings(BaseModel):
    model_config = _SECTION
    enabled: bool = False


class TelemetrySettings(BaseModel):
    model_config = _SECTION
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_logs: bool = False
    otel: bool = False


class BhootSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BHOOT_", env_nested_delimiter="__", extra="forbid", frozen=True
    )

    config_version: Literal[1] = 1
    home: Path | None = None  # runtime data dir; default <project>/.bhoot or ~/.cache/bhoot
    project_root: Path | None = None
    default_mode: Mode = Mode.BALANCED
    default_budget_usd: float | None = Field(default=0.25, ge=0)
    models: ModelTierConfig | None = None
    modes: Mapping[Mode, dict[str, Any]] = Field(default_factory=dict)  # per-mode overrides
    rate_limits: Mapping[str, RateLimit] = Field(default_factory=dict)  # key: model id
    vision: VisionSettings = Field(default_factory=VisionSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    packs: Mapping[str, dict[str, Any]] = Field(default_factory=dict)  # per-pack options
    debug: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        from bhoot.config.loader import YamlFileSource  # local import: loader imports settings

        return (init_settings, env_settings, YamlFileSource(settings_cls))

    def mode_profile(self, mode: Mode | None = None) -> ModeProfile:
        chosen = mode or self.default_mode
        return profile_for(chosen, self.modes.get(chosen))

    def runtime_home(self) -> Path:
        if self.home:
            return self.home
        if self.project_root:
            return self.project_root / ".bhoot"
        return Path.home() / ".cache" / "bhoot"
