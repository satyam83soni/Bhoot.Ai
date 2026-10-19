"""Prompting: system rules, safe template rendering, few-shot examples and message building."""

from bhoot.engine.prompting.examples import (
    EXAMPLE_SELECTORS,
    ByPageKindSelector,
    ExampleSelector,
    StaticSelector,
    TokenCappedSelector,
)
from bhoot.engine.prompting.messages import MessageBuilder, PromptInputs, neutralise, schema_summary
from bhoot.engine.prompting.system_prompts import (
    DEFAULT_PROMPT,
    GROUNDING_RULES,
    INJECTION_GUARD,
    OUTPUT_RULES,
)
from bhoot.engine.prompting.templates import (
    PROMPT_VARIABLES,
    PromptRenderError,
    check_prompt,
    render_prompt,
)

__all__ = [
    "DEFAULT_PROMPT",
    "EXAMPLE_SELECTORS",
    "GROUNDING_RULES",
    "INJECTION_GUARD",
    "OUTPUT_RULES",
    "PROMPT_VARIABLES",
    "ByPageKindSelector",
    "ExampleSelector",
    "MessageBuilder",
    "PromptInputs",
    "PromptRenderError",
    "StaticSelector",
    "TokenCappedSelector",
    "check_prompt",
    "neutralise",
    "render_prompt",
    "schema_summary",
]
