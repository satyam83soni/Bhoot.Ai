"""Nodes that work on field values: normalise, merge, validate, verify, decide, finalize."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from bhoot.domain.candidate import CandidateValue
from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import Document
from bhoot.domain.field_state import FieldState, FieldStatus
from bhoot.domain.modes import Tier
from bhoot.domain.result import JobStatus
from bhoot.domain.rules import RuleScope
from bhoot.engine.cost.pricing import CallShape, estimate_usd
from bhoot.engine.events import FieldDecided
from bhoot.engine.graph.state import Reset
from bhoot.engine.merge import Merger
from bhoot.engine.nodes.base import BaseNode, Update
from bhoot.engine.normalize.coercer import CoerceContext, TypeCoercer
from bhoot.engine.normalize.conventions import DocConventions, detect_conventions
from bhoot.engine.planning.repair import field_page
from bhoot.engine.policy import build_policy, decide_fields
from bhoot.engine.validation import ValidatorChain
from bhoot.engine.verification import EvidenceVerifier

_ROLE_FOR_TIER = {Tier.REPAIR: "cheap", Tier.STRONG: "strong", Tier.CROSS: "cross"}
REPAIR_PROMPT_CHARS = 3_000  # rules, instructions and feedback of a one-field repair call


class NormalizeNode(BaseNode):
    """Decide the document's conventions once (day-first dates, digit grouping), then read
    every new printed value into its type."""

    name = "normalize"

    async def run(self, state: Mapping[str, Any]) -> Update:
        schema = self.ctx.job.pack.schema
        candidates: tuple[CandidateValue, ...] = state.get("candidates", ())
        conventions: DocConventions | None = state.get("conventions")
        if conventions is None:
            dates, amounts = [], []
            for c in candidates:
                spec = schema.fields.get(c.path.pattern().raw)
                if spec is None or c.raw is None:
                    continue
                if spec.py_type == "date":
                    dates.append(str(c.raw))
                elif spec.py_type == "Decimal":
                    amounts.append(str(c.raw))
            conventions = detect_conventions(dates, amounts, self.ctx.job.pack.locale)
        ctx = CoerceContext(conventions, self.ctx.job.pack.locale)
        coercer = TypeCoercer()
        typed: list[CandidateValue] = []
        warnings: list[str] = []
        for c in candidates:
            spec = schema.fields.get(c.path.pattern().raw)
            if spec is None:
                warnings.append(f"dropped an answer for unknown field {c.path}")
                continue
            typed.append(coercer.coerce(c, spec, ctx))
        return {
            "candidates": Reset(tuple(typed)),
            "conventions": conventions,
            "warnings": tuple(warnings),
        }


class MergeNode(BaseNode):
    name = "merge"

    async def run(self, state: Mapping[str, Any]) -> Update:
        result = Merger().merge(
            schema=self.ctx.job.pack.schema,
            field_states=state.get("field_states", {}),
            candidates=state.get("candidates", ()),
            items=state.get("work_items", ()),
            nulls=state.get("nulls", ()),
            round_=int(state.get("round", 0)),
        )
        return {
            "field_states": result.field_states,
            "returned_null": tuple(sorted(result.returned_null)),
            "candidates": Reset(),
            "nulls": Reset(),
            "warnings": result.warnings,
        }


class ValidateNode(BaseNode):
    name = "validate"

    async def run(self, state: Mapping[str, Any]) -> Update:
        chain = ValidatorChain(self.ctx.rules(self.services.rules))
        result = chain.run(
            state.get("field_states", {}),
            locale=self.ctx.job.pack.locale,
            doc=state.get("document"),
            pack_options=self.ctx.pack_options,
        )
        return {"field_states": result.field_states, "doc_checks": result.doc_checks}


class VerifyNode(BaseNode):
    name = "verify"

    async def run(self, state: Mapping[str, Any]) -> Update:
        conventions: DocConventions | None = state.get("conventions")
        assert conventions is not None, "normalize runs before verify"
        states = EvidenceVerifier().verify(
            state.get("field_states", {}),
            state["document"],
            conventions,
            threshold=self.ctx.job.mode.fuzzy_quote_threshold,
        )
        return {"field_states": states}


class DecideNode(BaseNode):
    name = "decide"

    async def run(self, state: Mapping[str, Any]) -> Update:
        job, services = self.ctx.job, self.services
        doc: Document = state["document"]
        round_ = int(state.get("round", 0))
        states: dict[str, FieldState] = state.get("field_states", {})
        rules = self.ctx.rules(services.rules)

        def can_afford(decision: Decision) -> bool:
            if decision.target_tier is None or services.models is None:
                return True
            role = _ROLE_FOR_TIER.get(decision.target_tier, "cheap")
            spec = getattr(services.models, role, None) or services.models.cheap
            fs = states.get(decision.path.raw)
            page = field_page(fs) if fs is not None else None
            text = doc.page(page).text if page is not None and 0 <= page < doc.page_count else ""
            shape = CallShape(text_chars=len(text), prompt_chars=REPAIR_PROMPT_CHARS, leaves=1)
            estimate = estimate_usd(services.prices.price(spec.model_id), shape)
            return services.budget.can_spend(job.job_id, job.budget, estimate)

        result = decide_fields(
            states,
            doc=doc,
            profile=job.mode,
            round_=round_,
            policy=build_policy(services.settings.engine.policy_rules),
            returned_null=frozenset(state.get("returned_null", ())),
            has_cross_model=services.models is not None and services.models.cross is not None,
            can_afford=can_afford,
            critical_word_confidence=services.settings.ocr.critical_word_confidence,
            document_rule_ids=[r.rule_id for r in rules if r.scope is RuleScope.DOCUMENT],
        )
        for d in result.decisions:
            services.events.publish(
                FieldDecided(job.job_id, d.path.raw, d.kind, d.rule, d.reason, round_)
            )
        return {
            "field_states": result.field_states,
            "decisions": result.decisions,
            "round": round_ + 1,
            "warnings": result.warnings,
        }


class FinalizeNode(BaseNode):
    """Close every field that is still open (it goes to review) and set the job status."""

    name = "finalize"

    async def run(self, state: Mapping[str, Any]) -> Update:
        round_ = int(state.get("round", 0))
        states: dict[str, FieldState] = dict(state.get("field_states", {}))
        for path, fs in states.items():
            if not fs.is_terminal:
                states[path] = fs.decide(
                    DecisionKind.REVIEW, "unresolved at finalize", "finalize", round_
                )
        doc_failed = any(c.blocks for c in state.get("doc_checks", ()))
        review = doc_failed or any(fs.status is FieldStatus.NEEDS_REVIEW for fs in states.values())
        status = JobStatus.NEEDS_REVIEW if review else JobStatus.SUCCEEDED
        return {"field_states": states, "status": status.value}


def spent(state: Mapping[str, Any]) -> Decimal:
    cost = state.get("cost")
    return cost.total_usd if cost is not None else Decimal(0)
