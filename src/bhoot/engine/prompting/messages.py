"""Build the messages for one work item.

Order matters for cost: the system rules and the few-shot examples come first and are the same
for every page of a job, so providers' prompt caching can reuse them. The document comes last,
inside ``<document>`` tags, after the rendered (trusted) pack prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bhoot.domain.document import Document
from bhoot.domain.example import Example
from bhoot.domain.locales import Locale
from bhoot.domain.prompt import PromptTemplate
from bhoot.domain.schema import FieldSpec, SchemaDefinition
from bhoot.domain.work import WorkItem
from bhoot.engine.prompting.examples import ExampleSelector
from bhoot.engine.prompting.system_prompts import GROUNDING_RULES, INJECTION_GUARD, OUTPUT_RULES
from bhoot.engine.prompting.templates import render_prompt
from bhoot.engine.schema_compiler import SchemaDialect, SchemaTree, Selection
from bhoot.ports.llm import ContentPart, LLMMessage


def neutralise(text: str) -> str:
    """Stop page text from closing our tags early (a cheap guard against prompt injection)."""
    return text.replace("</document", "<\\/document").replace("</page", "<\\/page")


def schema_summary(schema: SchemaDefinition, item: WorkItem) -> str:
    tree = SchemaTree(schema)
    selection = Selection(tree, item.paths or None)
    lines: list[str] = []
    for spec in schema.fields.values():
        if spec.json_type in ("object", "array") or not selection.keeps(spec.path.parts()):
            continue
        lines.append(_describe(spec))
    return "\n".join(lines)


def _describe(spec: FieldSpec) -> str:
    text = f"- {spec.path.raw} ({spec.json_type}"
    text += ", required)" if spec.required else ")"
    if spec.description:
        text += f": {spec.description}"
    if spec.enum:
        text += f" [one of: {', '.join(spec.enum)}]"
    return text


@dataclass(frozen=True)
class PromptInputs:
    """What the builder needs from the job (kept small so tests don't need a whole job)."""

    schema: SchemaDefinition
    prompt: PromptTemplate
    examples: tuple[Example, ...] = ()
    locale: Locale = field(default_factory=Locale)
    pack_options: Mapping[str, Any] | None = None


class MessageBuilder:
    def __init__(self, examples: ExampleSelector) -> None:
        self.examples = examples

    def build(
        self,
        inputs: PromptInputs,
        item: WorkItem,
        doc: Document,
        *,
        envelope: Mapping[str, Any],
        dialect: SchemaDialect,
        crop_refs: tuple[str, ...] = (),
    ) -> tuple[LLMMessage, ...]:
        """``crop_refs`` are blob keys of the rendered ``item.crops`` (CROP items)."""
        system = "\n\n".join(
            part
            for part in (
                GROUNDING_RULES,
                INJECTION_GUARD,
                OUTPUT_RULES,
                dialect.instructions(envelope),
            )
            if part
        )
        messages: list[LLMMessage] = [LLMMessage.text("system", system)]
        kinds = [doc.page(i).kind for i in item.page_indices]
        for example in self.examples.select(
            inputs.examples, section=item.section, page_kinds=kinds
        ):
            messages.extend(_example_messages(example))
        prompt = render_prompt(
            inputs.prompt,
            {
                "schema_summary": schema_summary(inputs.schema, item),
                "section": item.section,
                "feedback": list(item.feedback),
                "kind": item.kind.value,
                "row_window": item.row_window,
                "locale": inputs.locale,
                "pack_options": dict(inputs.pack_options or {}),
            },
        )
        parts = _document_parts(prompt, item, doc, crop_refs)
        messages.append(LLMMessage(role="user", parts=parts))
        return tuple(messages)


def _document_parts(
    prompt: str, item: WorkItem, doc: Document, crop_refs: tuple[str, ...]
) -> tuple[ContentPart, ...]:
    parts: list[ContentPart] = []
    text = [prompt, "", "<document>"]

    def flush() -> None:
        if text:
            parts.append(ContentPart(type="text", text="\n".join(text)))
            text.clear()

    for index in item.page_indices:
        page = doc.page(index)
        if item.use_images and page.image_ref is not None:
            text.append(f'<page n="{page.number}" type="image"/>')
            flush()
            parts.append(ContentPart(type="image", image_ref=page.image_ref, detail="high"))
        else:
            text.append(f'<page n="{page.number}">\n{neutralise(page.text)}\n</page>')
    for k, crop in enumerate(crop_refs, start=1):
        text.append(f'<crop n="{k}"/>')
        flush()
        parts.append(ContentPart(type="image", image_ref=crop, detail="high"))
    text.append("</document>")
    flush()
    return tuple(parts)


def _example_messages(example: Example) -> Sequence[LLMMessage]:
    if example.input_image_ref is not None:
        user = LLMMessage(
            role="user",
            parts=(
                ContentPart(type="text", text='<document>\n<page n="1" type="image"/>'),
                ContentPart(type="image", image_ref=example.input_image_ref),
                ContentPart(type="text", text="</document>"),
            ),
        )
    else:
        body = neutralise(example.input_text or "")
        user = LLMMessage.text("user", f'<document>\n<page n="1">\n{body}\n</page>\n</document>')
    answer = json.dumps(example.output, ensure_ascii=False, default=str)
    return (user, LLMMessage.text("assistant", answer))
