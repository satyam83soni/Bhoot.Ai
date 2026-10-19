"""Resolve a request into a ``JobSpec``: pack, exact versions, mode, budget, options, cache key."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from bhoot._version import __version__
from bhoot.config.models import ModelSpec
from bhoot.config.settings import BhootSettings
from bhoot.domain.cost import Budget
from bhoot.domain.errors import RegistryError
from bhoot.domain.hashing import content_hash
from bhoot.domain.ids import new_job_id
from bhoot.domain.job import ExtractRequest, JobSpec
from bhoot.domain.pack import DocPack, SectionSpec
from bhoot.domain.schema import SchemaDefinition
from bhoot.domain.versions import ResolvedVersions
from bhoot.engine.prompting import DEFAULT_PROMPT, check_prompt


def ephemeral_pack(schema: SchemaDefinition) -> DocPack:
    """A schema passed directly: one section over every page, the built-in prompt."""
    return DocPack(
        name=schema.name,
        version=schema.version,
        schema=schema,
        prompt=DEFAULT_PROMPT,
        sections=(SectionSpec(name="all", paths=("*",)),),
    )


class JobFactory:
    def __init__(self, settings: BhootSettings, *, ocr_engine: str | None = None) -> None:
        self.settings = settings
        self.ocr_engine = ocr_engine

    def create(self, req: ExtractRequest, pack: DocPack | None = None) -> JobSpec:
        if pack is None:
            if not isinstance(req.schema, SchemaDefinition):
                raise RegistryError(
                    f"schema {req.schema!r} is a reference; pass a Schema class or a DocPack"
                )
            pack = ephemeral_pack(req.schema)
        check_prompt(pack.prompt)
        settings = self.settings
        mode = settings.mode_profile(req.mode)
        limit = req.max_cost_usd if req.max_cost_usd is not None else settings.default_budget_usd
        budget = Budget(Decimal(str(limit)) if limit is not None else None)
        pack_options = {
            **dict(pack.default_options),
            **settings.packs.get(pack.name, {}),
            **dict(req.options.pack_options),
        }
        models = {spec_role: spec.model_id for spec_role, spec in _roles(settings)}
        versions = ResolvedVersions(
            engine=__version__,
            schema_ref=pack.schema.ref,
            schema_hash=pack.schema.content_hash,
            prompt=pack.prompt.ref,
            prompt_hash=pack.prompt.content_hash,
            pack=pack.ref,
            rule_ids=tuple(sorted(r.rule_id for r in pack.rules)),
            models=models,
            ocr_engine=self.ocr_engine,
        )
        cache_key = content_hash(
            {
                "doc": req.doc_sha256,
                "versions": versions.model_dump(mode="json", by_alias=True),
                "mode": mode.model_dump(mode="json"),
                "options": req.options.cache_relevant(),
                "pack_options": pack_options,
            }
        )
        return JobSpec(
            job_id=new_job_id(),
            pack=pack,
            versions=versions,
            mode=mode,
            budget=budget,
            options=req.options,
            input_ref=req.input_ref,
            created_at=datetime.now(UTC),
            cache_key=cache_key,
            pack_options=pack_options,
        )


def _roles(settings: BhootSettings) -> list[tuple[str, ModelSpec]]:
    models = settings.models
    if models is None:
        return []
    out = []
    for role in ("cheap", "strong", "cross", "vision_cheap"):
        spec = getattr(models, role, None)
        if spec is not None:
            out.append((role, spec))
    return out
