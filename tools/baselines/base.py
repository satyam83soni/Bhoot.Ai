"""Baseline runners: the "before" numbers bhoot has to beat.

A runner turns one eval document into a predicted JSON dict. ``run_dataset`` loops over a
dataset, scores every case against expected.json and aggregates the metrics.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.baselines.documents import DocumentInput, load_document
from tools.baselines.scoring import Score, score_case


@dataclass(frozen=True)
class Prediction:
    data: dict[str, Any] | None
    cost_usd: float = 0.0
    error: str | None = None


class BaselineRunner(ABC):
    name: str

    @abstractmethod
    def predict(self, document: DocumentInput, schema_ref: str) -> Prediction: ...


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    tags: tuple[str, ...]
    score: Score
    cost_usd: float
    seconds: float
    error: str | None
    used_images: bool


def run_dataset(
    runner: BaselineRunner, dataset: Path, *, limit: int | None = None
) -> list[CaseResult]:
    manifest = yaml.safe_load((dataset / "manifest.yaml").read_text(encoding="utf-8"))
    schema_ref: str = manifest["schema"]
    results: list[CaseResult] = []
    for case in manifest["cases"][:limit]:
        case_dir = dataset / case["id"]
        expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
        document = load_document(case_dir / "document.pdf")
        started = time.perf_counter()
        prediction = runner.predict(document, schema_ref)
        results.append(CaseResult(
            case_id=case["id"],
            tags=tuple(case.get("tags", ())),
            score=score_case(expected, prediction.data),
            cost_usd=prediction.cost_usd,
            seconds=round(time.perf_counter() - started, 2),
            error=prediction.error,
            used_images=document.uses_images,
        ))  # fmt: skip
    return results


def summarise(runner: BaselineRunner, dataset: Path, results: list[CaseResult]) -> dict[str, Any]:
    total = Score()
    for result in results:
        total.add(result.score)
    return {
        "runner": runner.name,
        "dataset": dataset.as_posix(),
        "cases": len(results),
        "errors": sum(1 for r in results if r.error),
        "cost_usd_total": round(sum(r.cost_usd for r in results), 6),
        "cost_usd_per_doc": round(sum(r.cost_usd for r in results) / len(results), 6)
        if results
        else 0.0,
        "seconds_per_doc": round(sum(r.seconds for r in results) / len(results), 2)
        if results
        else 0.0,
        "metrics": total.as_dict(),
        "per_case": [
            {
                "id": r.case_id,
                "tags": list(r.tags),
                "error": r.error,
                "images": r.used_images,
                "cost_usd": r.cost_usd,
                "seconds": r.seconds,
                **r.score.as_dict(),
            }
            for r in results
        ],
    }
