"""Model prices and pre-call cost estimates.

Prices come only from ``bhoot.yaml`` (``input_cost_per_mtok`` / ``output_cost_per_mtok`` per
model). A model without prices costs $0, which also means budgets cannot stop it; ``bhoot
doctor`` warns about that.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bhoot.config.models import ModelTierConfig
from bhoot.domain.cost import ZERO, Usage

MILLION = Decimal(1_000_000)
CHARS_PER_TOKEN = 4  # rough, provider-neutral; only used for estimates
DEFAULT_IMAGE_TOKENS = 1_600  # a full page image at the resolution we send
OUTPUT_RATIO = Decimal("0.35")  # models rarely use the whole answer budget
TOKENS_PER_LEAF = 25  # value + quote + page for one scalar field
TOKENS_PER_ROW_FIELD = 12


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: Decimal
    output_per_mtok: Decimal

    def cost(self, usage: Usage) -> Decimal:
        billed_input = max(0, usage.prompt_tokens - usage.cached_tokens)
        usd = (
            billed_input * self.input_per_mtok + usage.completion_tokens * self.output_per_mtok
        ) / MILLION
        return usd.quantize(Decimal("0.00000001"))


FREE = Price(ZERO, ZERO)


class PriceBook:
    def __init__(self, models: ModelTierConfig | None) -> None:
        self._prices: dict[str, Price] = {}
        for spec in models.all_specs() if models else []:
            if spec.has_pricing:
                self._prices[spec.model_id] = Price(
                    Decimal(str(spec.input_cost_per_mtok)), Decimal(str(spec.output_cost_per_mtok))
                )

    def price(self, model_id: str) -> Price:
        return self._prices.get(model_id, FREE)

    def is_priced(self, model_id: str) -> bool:
        return model_id in self._prices


@dataclass(frozen=True, slots=True)
class CallShape:
    """What a work item will send and roughly get back, measured before the call."""

    text_chars: int = 0
    images: int = 0
    schema_chars: int = 0
    prompt_chars: int = 0  # system rules, instructions, examples, feedback
    leaves: int = 0  # scalar fields asked for
    expected_rows: int = 0  # rows expected in row arrays (upper bound is fine)
    row_fields: int = 0  # columns per row


def estimate_usage(shape: CallShape, *, image_tokens: int = DEFAULT_IMAGE_TOKENS) -> Usage:
    prompt = (shape.text_chars + shape.schema_chars + shape.prompt_chars) // CHARS_PER_TOKEN
    prompt += shape.images * image_tokens
    output = shape.leaves * TOKENS_PER_LEAF + shape.expected_rows * (
        shape.row_fields * TOKENS_PER_ROW_FIELD + TOKENS_PER_LEAF
    )
    completion = int(OUTPUT_RATIO * output) + 64
    return Usage(prompt_tokens=prompt, completion_tokens=completion, images=shape.images)


def estimate_usd(price: Price, shape: CallShape) -> Decimal:
    return price.cost(estimate_usage(shape))
