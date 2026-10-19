"""Checkpoint serializer for extraction state.

Every type that can sit in the graph state is registered under a stable name, so a checkpoint
written by one release loads in the next. Renaming a class must not change its name here.
Packs, jobs with callables and anything holding secrets never enter the state.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType

from bhoot.domain.candidate import CandidateRole, CandidateValue
from bhoot.domain.checks import CheckResult, CheckStatus, Severity
from bhoot.domain.cost import Budget, CostEntry, CostReport, Usage
from bhoot.domain.decisions import Decision, DecisionKind
from bhoot.domain.document import (
    BBox,
    CharBox,
    CropRequest,
    Document,
    Page,
    PageKind,
    Table,
    TableCell,
    Word,
)
from bhoot.domain.evidence import Evidence, EvidenceSource
from bhoot.domain.field_state import Attempt, FieldState, FieldStatus
from bhoot.domain.modes import Mode, Tier
from bhoot.domain.schema import Criticality, FieldPath, FieldSpec, Granularity
from bhoot.domain.work import WorkItem, WorkKind, WorkOutcome, WorkStatus
from bhoot.engine.graph.state import Reset
from bhoot.engine.normalize.conventions import DocConventions
from bhoot_graph import JsonSerializer

_DATACLASSES: dict[str, type] = {
    "FieldPath": FieldPath,
    "FieldSpec": FieldSpec,
    "BBox": BBox,
    "CharBox": CharBox,
    "Word": Word,
    "TableCell": TableCell,
    "Table": Table,
    "Page": Page,
    "Document": Document,
    "Evidence": Evidence,
    "CandidateValue": CandidateValue,
    "CheckResult": CheckResult,
    "Attempt": Attempt,
    "FieldState": FieldState,
    "Decision": Decision,
    "Usage": Usage,
    "CostEntry": CostEntry,
    "CostReport": CostReport,
    "Budget": Budget,
    "WorkItem": WorkItem,
    "WorkOutcome": WorkOutcome,
    "CropRequest": CropRequest,
    "DocConventions": DocConventions,
    "Reset": Reset,
}

_ENUMS: dict[str, type[Enum]] = {
    "CandidateRole": CandidateRole,
    "CheckStatus": CheckStatus,
    "Severity": Severity,
    "DecisionKind": DecisionKind,
    "PageKind": PageKind,
    "EvidenceSource": EvidenceSource,
    "FieldStatus": FieldStatus,
    "Mode": Mode,
    "Tier": Tier,
    "Criticality": Criticality,
    "Granularity": Granularity,
    "WorkKind": WorkKind,
    "WorkStatus": WorkStatus,
}


def engine_serializer() -> JsonSerializer:
    serde = JsonSerializer()
    serde.register(
        MappingProxyType, encode=dict, decode=MappingProxyType, name="bhoot.MappingProxy"
    )
    for name, cls in _ENUMS.items():
        serde.register_enum(cls, name=f"bhoot.{name}")
    for name, cls in _DATACLASSES.items():
        serde.register_dataclass(cls, name=f"bhoot.{name}")
    return serde
