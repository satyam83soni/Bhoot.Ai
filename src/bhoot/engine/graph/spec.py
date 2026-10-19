"""The extraction graph as data (``GraphSpec``) plus the registries that turn it into a
runnable graph. A pack or plugin can swap a node or insert one without editing the builder.

Default flow::

    load -> classify -> plan -> code_extract -> (llm_extract x N) -> normalize -> merge
         -> validate -> verify -> decide -> repair_plan -> (llm_extract x N) -> ...
                                         -> finalize
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from bhoot.domain.decisions import DecisionKind
from bhoot.domain.errors import GraphSpecError
from bhoot.domain.work import WorkItem
from bhoot.engine.nodes import (
    BaseNode,
    ClassifyNode,
    CodeExtractNode,
    DecideNode,
    FinalizeNode,
    LLMExtractNode,
    LoadNode,
    MergeNode,
    NormalizeNode,
    PlanNode,
    RepairPlanNode,
    ValidateNode,
    VerifyNode,
    light_page,
)
from bhoot.engine.registry import StrategyRegistry
from bhoot.engine.services import EngineServices, JobContext
from bhoot_graph import Send

NodeFactory = Callable[[EngineServices, JobContext], BaseNode]
Router = Callable[[Mapping[str, Any]], Any]
RouterFactory = Callable[[JobContext], Router]

SPENDING = frozenset({DecisionKind.REPAIR, DecisionKind.ESCALATE, DecisionKind.CROSS_CHECK})


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str | None = None  # a plain edge
    router: str | None = None  # or a router name with its possible targets
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphSpec:
    name: str
    version: str  # bump when saved state from older runs can't be resumed
    entry: str
    nodes: tuple[str, ...]
    edges: tuple[EdgeSpec, ...]
    finish: str

    def validate(
        self, factories: StrategyRegistry[NodeFactory], routers: StrategyRegistry[RouterFactory]
    ) -> None:
        known = set(self.nodes)
        for node in self.nodes:
            if node not in factories:
                raise GraphSpecError(f"graph {self.name}: no node factory named {node!r}")
        for edge in self.edges:
            ends = [edge.source, *([edge.target] if edge.target else []), *edge.targets]
            for end in ends:
                if end not in known:
                    raise GraphSpecError(f"graph {self.name}: edge refers to unknown node {end!r}")
            if edge.router is not None and edge.router not in routers:
                raise GraphSpecError(f"graph {self.name}: no router named {edge.router!r}")
            if (edge.target is None) == (edge.router is None):
                raise GraphSpecError(
                    f"graph {self.name}: edge from {edge.source} needs a target or a router"
                )
        if self.entry not in known or self.finish not in known:
            raise GraphSpecError(f"graph {self.name}: entry and finish must be nodes")

    def replace_node(self, old: str, new: str) -> GraphSpec:
        """Use another factory under a node's place in the flow."""

        def swap(name: str) -> str:
            return new if name == old else name

        edges = tuple(
            replace(e, source=swap(e.source), target=swap(e.target) if e.target else None,
                    targets=tuple(swap(t) for t in e.targets))
            for e in self.edges
        )  # fmt: skip
        return replace(self, nodes=tuple(swap(n) for n in self.nodes), edges=edges,
                       entry=swap(self.entry), finish=swap(self.finish))  # fmt: skip

    def insert_after(self, existing: str, new: str) -> GraphSpec:
        """Put ``new`` between ``existing`` and the node its plain edge pointed to."""
        edges = []
        found = False
        for e in self.edges:
            if e.source == existing and e.target is not None:
                edges.extend([EdgeSpec(existing, new), EdgeSpec(new, e.target)])
                found = True
            else:
                edges.append(e)
        if not found:
            raise GraphSpecError(f"{existing} has no plain edge to insert after")
        return replace(self, nodes=(*self.nodes, new), edges=tuple(edges))


# ---------------------------------------------------------------- routers
def route_work_items(ctx: JobContext) -> Router:
    """One ``llm_extract`` branch per pending work item, or straight on to ``normalize``."""

    def route(state: Mapping[str, Any]) -> Any:
        done = state.get("work_outcomes", {})
        pending: list[WorkItem] = [i for i in state.get("work_items", ()) if i.item_id not in done]
        if not pending:
            return "normalize"
        doc = state["document"]
        return [
            Send(
                "llm_extract",
                {"item": i, "pages": tuple(light_page(doc.page(p)) for p in i.page_indices)},
            )
            for i in pending
        ]

    return route


def route_after_decide(ctx: JobContext) -> Router:
    def route(state: Mapping[str, Any]) -> str:
        if int(state.get("round", 0)) >= ctx.job.mode.max_rounds:
            return "finalize"  # safety net; the last-round guard normally ends it first
        if any(d.kind in SPENDING for d in state.get("decisions", ())):
            return "repair_plan"
        return "finalize"

    return route


NODE_FACTORIES: StrategyRegistry[NodeFactory] = StrategyRegistry(
    "graph node",
    {
        "load": LoadNode,
        "classify": ClassifyNode,
        "plan": PlanNode,
        "code_extract": CodeExtractNode,
        "llm_extract": LLMExtractNode,
        "normalize": NormalizeNode,
        "merge": MergeNode,
        "validate": ValidateNode,
        "verify": VerifyNode,
        "decide": DecideNode,
        "repair_plan": RepairPlanNode,
        "finalize": FinalizeNode,
    },
)
ROUTERS: StrategyRegistry[RouterFactory] = StrategyRegistry(
    "graph router", {"work_items": route_work_items, "after_decide": route_after_decide}
)

DEFAULT_GRAPH = GraphSpec(
    name="bhoot.extraction",
    version="1",
    entry="load",
    nodes=(
        "load",
        "classify",
        "plan",
        "code_extract",
        "llm_extract",
        "normalize",
        "merge",
        "validate",
        "verify",
        "decide",
        "repair_plan",
        "finalize",
    ),
    edges=(
        EdgeSpec("load", "classify"),
        EdgeSpec("classify", "plan"),
        EdgeSpec("plan", "code_extract"),
        EdgeSpec("code_extract", router="work_items", targets=("llm_extract", "normalize")),
        EdgeSpec("llm_extract", "normalize"),
        EdgeSpec("normalize", "merge"),
        EdgeSpec("merge", "validate"),
        EdgeSpec("validate", "verify"),
        EdgeSpec("verify", "decide"),
        EdgeSpec("decide", router="after_decide", targets=("repair_plan", "finalize")),
        EdgeSpec("repair_plan", router="work_items", targets=("llm_extract", "normalize")),
    ),
    finish="finalize",
)
GRAPH_SPECS: StrategyRegistry[GraphSpec] = StrategyRegistry("graph", {"default": DEFAULT_GRAPH})
