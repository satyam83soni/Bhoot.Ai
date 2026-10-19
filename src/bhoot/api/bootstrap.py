"""Build the engine's services from settings: the composition root, the one place that picks
concrete adapters for each port.

LLM calls are wrapped (outermost first) in tracing, caching, a concurrency cap, per-model rate
limits and retries, so a cache hit skips everything below it and only real provider calls use
rate-limit budget.
"""

from __future__ import annotations

from bhoot.adapters.llm.decorators import (
    CachingLLM,
    ConcurrencyLimitedLLM,
    RateLimitedLLM,
    RetryingLLM,
    TracingLLM,
)
from bhoot.adapters.llm.rate_limiter import AsyncTokenBucket, SemaphoreLimiter
from bhoot.adapters.llm.router import ModelRouter
from bhoot.adapters.null import InMemoryLLMCache, NoopTracer, NullLLMCache
from bhoot.adapters.parsing.factory import MimeParserSelector
from bhoot.adapters.parsing.image_parser import ImageDocumentParser
from bhoot.adapters.parsing.images import (
    DefaultPageRenderer,
    PillowCropRenderer,
    PillowPreprocessor,
)
from bhoot.adapters.parsing.pdfium_parser import PdfiumDocumentParser
from bhoot.adapters.parsing.plumber_tables import PlumberTableExtractor
from bhoot.adapters.storage import MemoryBlobStore
from bhoot.config.settings import BhootSettings
from bhoot.domain.errors import ModelConfigError
from bhoot.engine.services import EngineServices
from bhoot.ports.llm import LLMClient, LLMRequest, LLMResponse, ModelCapabilities
from bhoot.ports.storage import BlobStore


class NoModelsConfigured:
    """Stands in when ``bhoot.yaml`` has no ``models:`` section: any call says how to fix it."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise ModelConfigError("no models configured: add a models: section to bhoot.yaml")

    def capabilities(self, model: str) -> ModelCapabilities:
        raise ModelConfigError("no models configured: add a models: section to bhoot.yaml")

    def count_tokens(self, model: str, text: str) -> int | None:
        return None


def build_llm(
    settings: BhootSettings, blobs: BlobStore, provider: LLMClient | None = None
) -> LLMClient:
    """The decorated LLM client. ``provider`` replaces the SDK router (custom or test clients)."""
    models = settings.models
    if provider is None:
        provider = ModelRouter(models, blobs) if models is not None else NoModelsConfigured()
    limits: dict[str, tuple[int | None, int | None]] = {}
    for spec in models.all_specs() if models else []:
        override = settings.rate_limits.get(spec.model_id)
        rpm = override.rpm if override and override.rpm else spec.rpm
        tpm = override.tpm if override and override.tpm else spec.tpm
        if rpm or tpm:
            limits[spec.model_id] = (rpm, tpm)
    retry = settings.retry
    client: LLMClient = RetryingLLM(
        provider, max_attempts=retry.max_attempts, initial_s=retry.initial_s, max_s=retry.max_s
    )
    client = RateLimitedLLM(client, AsyncTokenBucket(limits))
    client = ConcurrencyLimitedLLM(
        client, SemaphoreLimiter(settings.concurrency.max_parallel_llm_calls)
    )
    client = CachingLLM(client, InMemoryLLMCache() if settings.cache.llm else NullLLMCache())
    return TracingLLM(client, NoopTracer())


def build_services(settings: BhootSettings, *, llm: LLMClient | None = None) -> EngineServices:
    blobs = MemoryBlobStore()
    limits = settings.limits
    return EngineServices(
        settings=settings,
        llm=build_llm(settings, blobs, llm),
        blobs=blobs,
        parsers=MimeParserSelector([PdfiumDocumentParser(), ImageDocumentParser()]),
        renderer=DefaultPageRenderer(
            max_side_px=limits.max_render_side_px, max_image_pixels=limits.max_image_pixels
        ),
        preprocessor=PillowPreprocessor(),
        crops=PillowCropRenderer(),
        tables=PlumberTableExtractor(),
    )
