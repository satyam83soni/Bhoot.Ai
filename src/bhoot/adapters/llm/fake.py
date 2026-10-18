"""A scripted LLM for tests: answers are chosen by rules matched against the request metadata
(section, page, kind, tier, paths), so engine tests read like a story and need no network."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from bhoot.domain.cost import Usage
from bhoot.ports.llm import LLMRequest, LLMResponse, ModelCapabilities

Matcher = Callable[[LLMRequest], bool]


@dataclass
class _Rule:
    matches: Matcher
    answer: Mapping[str, Any] | Callable[[LLMRequest], Mapping[str, Any]] | Exception
    times: int | None = None  # None = unlimited
    used: int = 0


@dataclass
class _Builder:
    fake: FakeLLMClient
    matcher: Matcher

    def respond(
        self,
        answer: Mapping[str, Any] | Callable[[LLMRequest], Mapping[str, Any]],
        *,
        times: int | None = None,
    ) -> FakeLLMClient:
        self.fake.rules.append(_Rule(self.matcher, answer, times))
        return self.fake

    def fail(self, error: Exception, *, times: int | None = None) -> FakeLLMClient:
        self.fake.rules.append(_Rule(self.matcher, error, times))
        return self.fake


@dataclass
class FakeLLMClient:
    vision: bool = True
    cost_per_call: Decimal = Decimal("0.001")
    rules: list[_Rule] = field(default_factory=list)
    calls: list[LLMRequest] = field(default_factory=list)

    def when(self, **meta: str | int) -> _Builder:
        """Match on request metadata, e.g. when(section="summary", kind="repair")."""
        expected = {k: str(v) for k, v in meta.items()}

        def matcher(request: LLMRequest) -> bool:
            wanted_path = expected.get("paths_contains")
            if wanted_path is not None and wanted_path not in request.metadata.get("paths", ""):
                return False
            return all(
                request.metadata.get(k) == v for k, v in expected.items() if k != "paths_contains"
            )

        return _Builder(self, matcher)

    def when_request(self, matcher: Matcher) -> _Builder:
        return _Builder(self, matcher)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        for rule in self.rules:
            if (rule.times is None or rule.used < rule.times) and rule.matches(request):
                rule.used += 1
                if isinstance(rule.answer, Exception):
                    raise rule.answer
                payload = rule.answer(request) if callable(rule.answer) else rule.answer
                content = json.dumps(payload, default=str)
                return LLMResponse(
                    content=content,
                    model=request.model,
                    parsed=dict(payload),
                    usage=Usage(prompt_tokens=100, completion_tokens=len(content) // 4),
                    cost_usd=self.cost_per_call,
                    latency_ms=1,
                )
        raise AssertionError(
            f"FakeLLMClient: no rule matches request with metadata {dict(request.metadata)}"
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(
            vision=self.vision, json_schema=True, strict_schema=True, dialect="openai_strict"
        )

    def count_tokens(self, model: str, text: str) -> int | None:
        return len(text) // 4

    def assert_called(self, *, times: int) -> None:
        assert len(self.calls) == times, f"expected {times} LLM calls, got {len(self.calls)}"
