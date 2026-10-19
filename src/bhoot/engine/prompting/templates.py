"""Render pack prompts safely.

Templates run in Jinja's sandbox with strict undefined variables, and only the documented
variables are passed in. Document text is never rendered as a template: it is appended
afterwards as data, so a statement that contains ``{{ ... }}`` can't execute anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment

from bhoot.domain.errors import ConfigError
from bhoot.domain.prompt import PromptTemplate

# what a pack prompt may use
PROMPT_VARIABLES = frozenset(
    {"schema_summary", "section", "feedback", "kind", "row_window", "locale", "pack_options"}
)

_ENV = SandboxedEnvironment(
    undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False
)


class PromptRenderError(ConfigError):
    """A pack prompt uses an unknown variable or has a template error."""


@lru_cache(maxsize=256)
def _compile(body: str) -> Any:
    return _ENV.from_string(body)


def check_prompt(prompt: PromptTemplate) -> None:
    """Fail at pack load, not mid-job, when a prompt declares variables we never provide."""
    unknown = prompt.variables - PROMPT_VARIABLES
    if unknown:
        raise PromptRenderError(
            f"prompt {prompt.ref} uses unknown variables {sorted(unknown)}; "
            f"available: {sorted(PROMPT_VARIABLES)}"
        )
    try:
        _compile(prompt.body)
    except TemplateError as exc:
        raise PromptRenderError(f"prompt {prompt.ref}: {exc}") from exc


def render_prompt(prompt: PromptTemplate, context: Mapping[str, Any]) -> str:
    values = {key: context.get(key) for key in PROMPT_VARIABLES}
    try:
        return str(_compile(prompt.body).render(**values)).strip()
    except TemplateError as exc:
        raise PromptRenderError(f"prompt {prompt.ref}: {exc}") from exc
