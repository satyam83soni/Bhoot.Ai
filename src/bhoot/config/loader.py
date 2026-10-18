"""Find, read and validate ``bhoot.yaml``.

Discovery order: explicit path → ``BHOOT_CONFIG`` env var → ``./bhoot.yaml`` → ``bhoot.yaml``
in the project root (nearest folder with ``pyproject.toml`` or ``bhoot.yaml``) → no file.
Precedence of values: code arguments > ``BHOOT_*`` env vars > the file > defaults.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource

from bhoot.config.settings import BhootSettings
from bhoot.domain.errors import ConfigFileError

CONFIG_NAME = "bhoot.yaml"
DEFINITION_KEYS = frozenset({"schemas", "prompts", "examples", "rules", "definitions"})

_FILE_VALUES: ContextVar[Mapping[str, Any] | None] = ContextVar("bhoot_file_values", default=None)


class YamlFileSource(PydanticBaseSettingsSource):
    """Settings source holding the parsed bhoot.yaml (lower priority than env vars)."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        values = _FILE_VALUES.get() or {}
        return values.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(_FILE_VALUES.get() or {})


@dataclass(frozen=True)
class LoadedConfig:
    settings: BhootSettings
    path: Path | None  # the bhoot.yaml that was used, if any


def find_project_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for folder in (current, *current.parents):
        if (folder / CONFIG_NAME).is_file() or (folder / "pyproject.toml").is_file():
            return folder
    return None


def find_config_file(explicit: str | Path | None, project_root: Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ConfigFileError(f"config file {path} does not exist")
        return path
    env = os.environ.get("BHOOT_CONFIG")
    if env:
        path = Path(env)
        if not path.is_file():
            raise ConfigFileError(f"BHOOT_CONFIG points to {path}, which does not exist")
        return path
    for candidate in (
        Path.cwd() / CONFIG_NAME,
        project_root / CONFIG_NAME if project_root else None,
    ):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def read_config_file(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigFileError(f"{path} must contain a mapping of settings at the top level")
    misplaced = DEFINITION_KEYS & set(data)
    if misplaced:
        raise ConfigFileError(
            f"{path}: '{sorted(misplaced)[0]}' does not belong in bhoot.yaml. Define schemas, "
            "prompts, examples and rules in code (bhoot.define) and version them with "
            "`bhoot migrate dev`."
        )
    return data


@contextmanager
def _file_values(values: Mapping[str, Any]) -> Iterator[None]:
    token = _FILE_VALUES.set(values)
    try:
        yield
    finally:
        _FILE_VALUES.reset(token)


def load_config(
    config: str | Path | None = None, *, project_dir: str | Path | None = None, **overrides: Any
) -> LoadedConfig:
    root = find_project_root(Path(project_dir) if project_dir else None)
    path = find_config_file(config, root)
    values = read_config_file(path) if path else {}
    overrides.setdefault("project_root", root)
    with _file_values(values):
        try:
            settings = BhootSettings(**overrides)
        except ValidationError as exc:
            raise ConfigFileError(_explain(exc, path)) from exc
    return LoadedConfig(settings, path)


def load_settings(
    config: str | Path | None = None, *, project_dir: str | Path | None = None, **overrides: Any
) -> BhootSettings:
    return load_config(config, project_dir=project_dir, **overrides).settings


# -------------------------------------------------------------------- friendly errors
def _explain(exc: ValidationError, path: Path | None) -> str:
    where = str(path) if path else "settings"
    problems = []
    for err in exc.errors():
        loc = [str(part) for part in err["loc"]]
        dotted = ".".join(loc)
        if err["type"] == "extra_forbidden":
            hint = _suggest(loc)
            problems.append(
                f"{dotted} is not a setting" + (f"; did you mean {hint}?" if hint else "")
            )
        else:
            problems.append(f"{dotted}: {err['msg']}")
    return f"{where}: " + "; ".join(problems)


def _suggest(loc: list[str]) -> str | None:
    model: type[BaseModel] | None = BhootSettings
    for part in loc[:-1]:
        field = model.model_fields.get(part) if model else None
        annotation = field.annotation if field else None
        model = (
            annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel)
            else None
        )
    if model is None:
        return None
    match = difflib.get_close_matches(loc[-1], list(model.model_fields), n=1, cutoff=0.6)
    return ".".join([*loc[:-1], match[0]]) if match else None
