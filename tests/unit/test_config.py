"""Configuration: discovery, precedence, validation messages, modes and model tiers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bhoot.config import BhootSettings, ModelSpec, ModelTierConfig, load_config, load_settings
from bhoot.config.modes import MODE_PROFILES
from bhoot.domain.errors import ConfigFileError, ModelConfigError
from bhoot.domain.modes import Mode, Tier
from bhoot.domain.schema import Criticality


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='app'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key in ("BHOOT_CONFIG", "BHOOT_DEFAULT_MODE", "BHOOT_OCR__DPI", "BHOOT_DEFAULT_BUDGET_USD"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def _write(folder: Path, text: str, name: str = "bhoot.yaml") -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_any_file(project: Path) -> None:
    loaded = load_config()
    assert loaded.path is None
    s = loaded.settings
    assert s.default_mode is Mode.BALANCED
    assert s.ocr.dpi == 300
    assert s.models is None
    assert s.project_root == project.resolve()
    assert s.runtime_home() == project.resolve() / ".bhoot"


def test_file_is_found_and_read(project: Path) -> None:
    _write(
        project, "config_version: 1\ndefault_mode: economy\nocr:\n  preprocess: off\n  dpi: 250\n"
    )
    loaded = load_config()
    assert loaded.path == project / "bhoot.yaml"
    assert loaded.settings.default_mode is Mode.ECONOMY
    assert loaded.settings.ocr.preprocess == "off"
    assert loaded.settings.ocr.dpi == 250


def test_precedence_code_over_env_over_file(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(project, "default_mode: economy\nocr:\n  dpi: 250\n")
    monkeypatch.setenv("BHOOT_DEFAULT_MODE", "max_accuracy")
    monkeypatch.setenv("BHOOT_OCR__DPI", "320")
    s = load_settings()
    assert s.default_mode is Mode.MAX_ACCURACY  # env beats file
    assert s.ocr.dpi == 320
    s = load_settings(default_mode="balanced")
    assert s.default_mode is Mode.BALANCED  # code beats env


def test_explicit_path_and_env_pointer(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other = _write(project, "default_mode: economy\n", name="prod.yaml")
    assert load_settings(other).default_mode is Mode.ECONOMY
    monkeypatch.setenv("BHOOT_CONFIG", str(other))
    assert load_config().path == other
    monkeypatch.setenv("BHOOT_CONFIG", str(project / "missing.yaml"))
    with pytest.raises(ConfigFileError, match="does not exist"):
        load_config()


def test_typos_get_suggestions(project: Path) -> None:
    _write(project, "ocr:\n  preproces: auto\n")
    with pytest.raises(
        ConfigFileError, match=r"ocr\.preproces is not a setting; did you mean ocr\.preprocess"
    ):
        load_config()
    _write(project, "defualt_mode: economy\n")
    with pytest.raises(ConfigFileError, match="did you mean default_mode"):
        load_config()


def test_bad_values_are_reported(project: Path) -> None:
    _write(project, "ocr:\n  dpi: 5\n")
    with pytest.raises(ConfigFileError, match=r"ocr\.dpi"):
        load_config()


@pytest.mark.parametrize("key", ["schemas", "prompts", "rules"])
def test_definitions_do_not_belong_in_the_file(project: Path, key: str) -> None:
    _write(project, f"{key}:\n  x: 1\n")
    with pytest.raises(ConfigFileError, match=r"does not belong in bhoot\.yaml"):
        load_config()


def test_invalid_yaml_and_non_mapping(project: Path) -> None:
    _write(project, "ocr: [unclosed\n")
    with pytest.raises(ConfigFileError, match="not valid YAML"):
        load_config()
    _write(project, "- just\n- a list\n")
    with pytest.raises(ConfigFileError, match="mapping"):
        load_config()
    _write(project, "")
    assert load_config().settings.default_mode is Mode.BALANCED


# ---------------------------------------------------------------- modes
def test_builtin_profiles_follow_the_design() -> None:
    balanced = MODE_PROFILES[Mode.BALANCED]
    assert balanced.max_tier(Criticality.CRITICAL) is Tier.CROSS
    assert balanced.max_tier(Criticality.NORMAL) is Tier.REPAIR
    assert MODE_PROFILES[Mode.ECONOMY].cross_check_threshold is None
    assert MODE_PROFILES[Mode.MAX_ACCURACY].max_rounds == 6


def test_mode_overrides_from_config(project: Path) -> None:
    _write(project, "modes:\n  balanced:\n    cross_check_threshold: 0.85\n")
    s = load_settings()
    assert s.mode_profile().cross_check_threshold == 0.85
    assert s.mode_profile(Mode.ECONOMY).cross_check_threshold is None


# ---------------------------------------------------------------- models
def _tiers(**kw: ModelSpec | None) -> ModelTierConfig:
    args = {"cheap": ModelSpec(model="openai/cheap"), "strong": ModelSpec(model="anthropic/strong")}
    args.update(kw)  # type: ignore[arg-type]
    return ModelTierConfig(**args)  # type: ignore[arg-type]


def test_model_resolution_and_vision_fallback() -> None:
    blind = {"openai/cheap": False, "anthropic/strong": True, "openai/eyes": True}
    sees = blind.__getitem__
    tiers = _tiers(vision_cheap=ModelSpec(model="openai/eyes"))
    assert tiers.resolve("cheap", needs_vision=False, has_vision=sees).model == "openai/cheap"
    assert tiers.resolve("cheap", needs_vision=True, has_vision=sees).model == "openai/eyes"
    assert tiers.resolve("strong", needs_vision=True, has_vision=sees).model == "anthropic/strong"
    with pytest.raises(ModelConfigError, match="cannot read images"):
        _tiers().resolve("cheap", needs_vision=True, has_vision=sees)
    with pytest.raises(ModelConfigError, match="no 'cross' model"):
        _tiers().resolve("cross", needs_vision=False, has_vision=sees)
    forced = _tiers(
        cheap=ModelSpec(model="x", provider="openai", api_base="http://gpu:8000/v1", vision=True)
    )
    assert (
        forced.resolve("cheap", needs_vision=True, has_vision=lambda _: False).model_id
        == "openai/x"
    )


def test_models_from_file_and_settings_are_frozen(project: Path) -> None:
    _write(
        project,
        "models:\n  cheap: {model: openai/c, api_key_env: OPENAI_API_KEY}\n"
        "  strong: {model: anthropic/s}\nrate_limits:\n  openai/c: {rpm: 500}\n",
    )
    s = load_settings()
    assert s.models is not None
    assert s.models.cheap.api_key_env == "OPENAI_API_KEY"
    assert s.rate_limits["openai/c"].rpm == 500
    with pytest.raises(Exception, match="frozen"):
        s.debug = True  # type: ignore[misc]
    assert isinstance(s, BhootSettings)
