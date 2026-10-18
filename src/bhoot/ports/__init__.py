"""Ports: Protocol interfaces for everything external (LLM, parsing, OCR, registry, caches, storage, tracing).

The engine depends only on these. Import them from here: ``from bhoot.ports import LLMClient``.
"""

from bhoot.ports.llm import (
    ContentPart,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ModelCapabilities,
    ResponseFormat,
)
from bhoot.ports.ocr import (
    AssembledText,
    OcrEngine,
    OcrLine,
    OcrPage,
    OcrTextAssembler,
    OcrWord,
    PreparedImage,
    ScanPreprocessor,
)
from bhoot.ports.parsing import (
    CropRenderer,
    DocumentInput,
    DocumentParser,
    ImagePreprocessor,
    PageRenderer,
    ParserSelector,
    TableExtractor,
)
from bhoot.ports.storage import (
    BlobStore,
    CheckpointerFactory,
    Clock,
    DebugSink,
    LLMCache,
    Registry,
    ResultCache,
    Span,
    Tracer,
    TrainingSink,
)

__all__ = [
    "AssembledText",
    "BlobStore",
    "CheckpointerFactory",
    "Clock",
    "ContentPart",
    "CropRenderer",
    "DebugSink",
    "DocumentInput",
    "DocumentParser",
    "ImagePreprocessor",
    "LLMCache",
    "LLMClient",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "ModelCapabilities",
    "OcrEngine",
    "OcrLine",
    "OcrPage",
    "OcrTextAssembler",
    "OcrWord",
    "PageRenderer",
    "ParserSelector",
    "PreparedImage",
    "Registry",
    "ResponseFormat",
    "ResultCache",
    "ScanPreprocessor",
    "Span",
    "TableExtractor",
    "Tracer",
    "TrainingSink",
]
