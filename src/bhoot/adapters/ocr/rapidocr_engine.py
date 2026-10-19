"""Local OCR with RapidOCR (PP-OCR models on onnxruntime, CPU). Installed with ``bhoot[ocr]``.

The ``mobile`` profile uses the small models bundled in the package, so nothing is downloaded
and no page leaves the machine. ``server`` uses the larger models, downloaded once on first
use. The engine object is not documented as thread-safe, so calls are serialised.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import threading
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from bhoot.domain.document import BBox
from bhoot.domain.errors import ConfigError
from bhoot.ports.ocr import OcrLine, OcrPage, OcrWord

MODEL_TYPES = {"mobile": "small", "server": "medium"}  # PP-OCRv6 models (rapidocr >= 3.9)


def _quad_box(quad: Any) -> BBox:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _split_line(text: str, box: BBox, score: float) -> tuple[OcrWord, ...]:
    """Word boxes from a line box when the engine gives none: proportional to text length."""
    words = text.split()
    if not words:
        return ()
    total = sum(len(w) for w in words) + len(words) - 1
    per_char = box.width / max(1, total)
    out, x = [], box.x0
    for word in words:
        width = per_char * len(word)
        out.append(OcrWord(word, BBox(x, box.y0, x + width, box.y1), score))
        x += width + per_char
    return tuple(out)


def _supported(params: dict[str, Any]) -> dict[str, Any]:
    """Only the settings the installed RapidOCR release knows (older ones reject new keys)."""
    import rapidocr
    import yaml

    config_file = Path(rapidocr.__file__).parent / "config.yaml"
    try:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}

    def known(key: str) -> bool:
        node: Any = config
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    return {k: v for k, v in params.items() if known(k)}


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(
        self, profile: Literal["mobile", "server"] = "mobile", *, threads: int = 2
    ) -> None:
        if profile not in MODEL_TYPES:
            raise ConfigError(f"unknown OCR model profile {profile!r}; use mobile or server")
        self.profile = profile
        self.threads = threads
        self._engine: Any = None
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        try:
            return importlib.metadata.version("rapidocr")
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    @property
    def engine_id(self) -> str:
        return f"rapidocr-{self.version}-{self.profile}"

    def available(self) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in ("rapidocr", "onnxruntime"))

    def _load(self) -> Any:
        if self._engine is None:
            if not self.available():
                raise ConfigError("OCR needs its extra: pip install 'bhoot[ocr]'")
            from rapidocr import ModelType, RapidOCR

            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            model_type = ModelType(MODEL_TYPES[self.profile])
            wanted = {
                "Global.log_level": "error",
                "Det.model_type": model_type,
                "Rec.model_type": model_type,
                "EngineConfig.onnxruntime.intra_op_num_threads": self.threads,
            }
            self._engine = RapidOCR(params=_supported(wanted))
        return self._engine

    def recognize(self, png: bytes, *, lang: Sequence[str] = ("en",)) -> OcrPage:
        with Image.open(BytesIO(png)) as img:
            width, height = img.size
        with self._lock:
            out = self._load()(png, return_word_box=True)
        boxes = out.boxes if out.boxes is not None else []
        texts = out.txts or ()
        scores = out.scores or ()
        word_results = out.word_results or ()
        lines: list[OcrLine] = []
        for i, (quad, text, score) in enumerate(zip(boxes, texts, scores, strict=False)):
            box = _quad_box(quad)
            words: tuple[OcrWord, ...] = ()
            if i < len(word_results) and word_results[i]:
                words = tuple(
                    OcrWord(str(w[0]), _quad_box(w[2]), float(w[1]))
                    for w in word_results[i]
                    if str(w[0]).strip()
                )
            if not words:
                words = _split_line(str(text), box, float(score))
            if words:
                lines.append(OcrLine(words, box))
        lines.sort(key=lambda line: (line.bbox.y0, line.bbox.x0))
        return OcrPage(tuple(lines), width, height, self.engine_id)
