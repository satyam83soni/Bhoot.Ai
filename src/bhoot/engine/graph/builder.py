"""Turn a ``GraphSpec`` into a runnable ``bhoot_graph`` graph for one job."""

from __future__ import annotations

from bhoot.engine.graph.spec import (
    DEFAULT_GRAPH,
    NODE_FACTORIES,
    ROUTERS,
    GraphSpec,
    NodeFactory,
    RouterFactory,
)
from bhoot.engine.graph.state import ExtractionState
from bhoot.engine.registry import StrategyRegistry
from bhoot.engine.services import EngineServices, JobContext
from bhoot_graph import END, START, BaseCheckpointSaver, CompiledGraph, StateGraph


def build_graph(
    services: EngineServices,
    ctx: JobContext,
    *,
    spec: GraphSpec = DEFAULT_GRAPH,
    checkpointer: BaseCheckpointSaver | None = None,
    factories: StrategyRegistry[NodeFactory] = NODE_FACTORIES,
    routers: StrategyRegistry[RouterFactory] = ROUTERS,
) -> CompiledGraph:
    spec.validate(factories, routers)
    graph = StateGraph(ExtractionState, name=spec.name)
    for name in spec.nodes:
        graph.add_node(name, factories.get(name)(services, ctx))
    graph.add_edge(START, spec.entry)
    for edge in spec.edges:
        if edge.target is not None:
            graph.add_edge(edge.source, edge.target)
        else:
            assert edge.router is not None
            graph.add_conditional_edges(
                edge.source, routers.get(edge.router)(ctx), list(edge.targets)
            )
    graph.add_edge(spec.finish, END)
    return graph.compile(checkpointer, version=spec.version)


def recursion_limit(max_rounds: int) -> int:
    """Supersteps a job may take: the first pass plus about seven steps per repair round."""
    return 25 + 10 * max_rounds
