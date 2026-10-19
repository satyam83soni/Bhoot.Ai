"""``StateGraph``: declare nodes and edges, then ``compile()`` into a runnable graph.

The API follows LangGraph's so the same code reads the same way::

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("extract", extract)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route, ["extract", END])
    g.add_edge("extract", END)
    app = g.compile()
    final = await app.ainvoke({"pages": 3})
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from bhoot_graph.constants import END, RESERVED, START
from bhoot_graph.errors import InvalidGraphError
from bhoot_graph.state import Channel, channels_for

if TYPE_CHECKING:
    from bhoot_graph.checkpoint.base import BaseCheckpointSaver
    from bhoot_graph.runner import CompiledGraph

NodeAction = Callable[..., Any]
Router = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class NodeSpec:
    name: str
    action: Any  # a callable, or a CompiledGraph used as a subgraph
    is_async: bool
    takes_config: bool
    destinations: tuple[str, ...] = ()  # where a Command(goto=...) from this node may go


@dataclass(frozen=True, slots=True)
class Branch:
    path: Router
    path_map: Mapping[Hashable, str] | None  # router result -> node; None = names as-is
    is_async: bool
    takes_config: bool


def _signature_info(fn: Callable[..., Any]) -> tuple[bool, bool]:
    target = fn if inspect.isfunction(fn) or inspect.ismethod(fn) else type(fn).__call__
    is_async = inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(target)
    try:
        takes_config = "config" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins without a signature
        takes_config = False
    return is_async, takes_config


class StateGraph:
    def __init__(self, state_schema: type, *, name: str | None = None) -> None:
        self.state_schema = state_schema
        self.name = name or state_schema.__name__
        self.channels: dict[str, Channel] = channels_for(state_schema)
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: list[tuple[str, str]] = []
        self.waiting_edges: list[tuple[tuple[str, ...], str]] = []
        self.branches: dict[str, list[Branch]] = {}

    # ------------------------------------------------------------------ nodes
    def add_node(
        self,
        node: str | NodeAction,
        action: NodeAction | None = None,
        *,
        destinations: Sequence[str] = (),
    ) -> Self:
        """Add a node. ``add_node(fn)`` names it after the function.

        A node receives the state (or a ``Send`` payload), plus ``config`` if it has a
        parameter named so, and returns a partial update, a ``Command`` or None. A compiled
        graph can be a node too: it gets the keys both states share and returns its result.
        """
        if action is None:
            if isinstance(node, str):
                raise InvalidGraphError(f"node {node!r} needs an action")
            action, name = node, getattr(node, "__name__", "")
        elif isinstance(node, str):
            name = node
        else:
            raise InvalidGraphError("add_node(name, action): name must be a string")
        if not name or name in RESERVED:
            raise InvalidGraphError(f"invalid node name {name!r}")
        if name in self.nodes:
            raise InvalidGraphError(f"node {name!r} already exists")
        if name in self.channels:
            raise InvalidGraphError(f"node {name!r} has the same name as a state key")
        from bhoot_graph.runner import CompiledGraph

        if isinstance(action, CompiledGraph):
            is_async, takes_config = True, True
        elif callable(action):
            is_async, takes_config = _signature_info(action)
        else:
            raise InvalidGraphError(f"node {name!r}: action must be callable")
        self.nodes[name] = NodeSpec(name, action, is_async, takes_config, tuple(destinations))
        return self

    def add_sequence(self, nodes: Sequence[NodeAction | tuple[str, NodeAction]]) -> Self:
        """Add nodes and chain them with edges in the given order."""
        if not nodes:
            raise InvalidGraphError("add_sequence needs at least one node")
        names: list[str] = []
        for item in nodes:
            if isinstance(item, tuple):
                self.add_node(item[0], item[1])
                names.append(item[0])
            else:
                self.add_node(item)
                names.append(getattr(item, "__name__", ""))
        for a, b in itertools.pairwise(names):
            self.add_edge(a, b)
        return self

    # ------------------------------------------------------------------ edges
    def add_edge(self, start_key: str | Sequence[str], end_key: str) -> Self:
        """``a -> b``: b runs after a. With a list of starts, b waits until all have run."""
        if end_key == START:
            raise InvalidGraphError("START cannot be an edge target")
        if isinstance(start_key, str):
            if start_key == END:
                raise InvalidGraphError("END cannot be an edge source")
            self.edges.append((start_key, end_key))
            return self
        starts = tuple(start_key)
        if not starts or START in starts or END in starts:
            raise InvalidGraphError(f"invalid waiting edge sources {starts!r}")
        self.waiting_edges.append((starts, end_key))
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Router,
        path_map: Mapping[Hashable, str] | Sequence[str] | None = None,
    ) -> Self:
        """After ``source``, call ``path(state)`` to pick the next node(s).

        ``path`` returns a node name, a ``Send``, or a list of them (``[]`` goes nowhere).
        ``path_map`` maps its return values to node names, or lists the possible names so
        ``compile()`` can check them.
        """
        if source == END:
            raise InvalidGraphError("END cannot be an edge source")
        mapping: Mapping[Hashable, str] | None
        if path_map is None:
            mapping = None
        elif isinstance(path_map, Mapping):
            mapping = dict(path_map)
        else:
            mapping = {name: name for name in path_map}
        is_async, takes_config = _signature_info(path)
        self.branches.setdefault(source, []).append(Branch(path, mapping, is_async, takes_config))
        return self

    def set_entry_point(self, key: str) -> Self:
        return self.add_edge(START, key)

    def set_conditional_entry_point(
        self, path: Router, path_map: Mapping[Hashable, str] | Sequence[str] | None = None
    ) -> Self:
        return self.add_conditional_edges(START, path, path_map)

    def set_finish_point(self, key: str) -> Self:
        return self.add_edge(key, END)

    # ------------------------------------------------------------------ compile
    def compile(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        *,
        name: str | None = None,
        version: str | None = None,
    ) -> CompiledGraph:
        """Check the wiring and return a runnable graph (a snapshot of this builder).

        With a ``checkpointer`` every superstep is saved under the run's ``thread_id``.
        ``version`` pins saved runs: a thread saved by another version is not resumed.
        """
        self.validate()
        from bhoot_graph.runner import CompiledGraph

        return CompiledGraph(
            name=name or self.name,
            channels=dict(self.channels),
            nodes=dict(self.nodes),
            edges=tuple(self.edges),
            waiting_edges=tuple(self.waiting_edges),
            branches={k: tuple(v) for k, v in self.branches.items()},
            checkpointer=checkpointer,
            version=version,
        )

    def validate(self) -> None:
        known = set(self.nodes)

        def check(name: str, where: str, *, target: bool) -> None:
            allowed = known | ({END} if target else {START})
            if name not in allowed:
                raise InvalidGraphError(f"{where} refers to unknown node {name!r}")

        for a, b in self.edges:
            check(a, f"edge {a} -> {b}", target=False)
            check(b, f"edge {a} -> {b}", target=True)
        for starts, b in self.waiting_edges:
            for a in starts:
                check(a, f"edge {list(starts)} -> {b}", target=False)
            check(b, f"edge {list(starts)} -> {b}", target=True)
        for source, branches in self.branches.items():
            check(source, f"conditional edge from {source}", target=False)
            for branch in branches:
                for target in (branch.path_map or {}).values():
                    check(target, f"conditional edge from {source}", target=True)
        for spec in self.nodes.values():
            for target in spec.destinations:
                check(target, f"destinations of {spec.name}", target=True)
        if not any(a == START for a, _ in self.edges) and START not in self.branches:
            raise InvalidGraphError("the graph has no entry point: add an edge from START")
        unreachable = known - self._reachable()
        if unreachable:
            raise InvalidGraphError(
                f"nodes never reached from START: {sorted(unreachable)} "
                "(add an edge, a path_map, or destinations=[...] for Command gotos)"
            )

    def _reachable(self) -> set[str]:
        """Nodes reachable from START. A router without a path_map may go anywhere."""
        if any(b.path_map is None for bs in self.branches.values() for b in bs):
            return set(self.nodes)
        targets: dict[str, set[str]] = {}
        for a, b in self.edges:
            targets.setdefault(a, set()).add(b)
        for starts, b in self.waiting_edges:
            for a in starts:
                targets.setdefault(a, set()).add(b)
        for source, branches in self.branches.items():
            for branch in branches:
                targets.setdefault(source, set()).update((branch.path_map or {}).values())
        for spec in self.nodes.values():
            targets.setdefault(spec.name, set()).update(spec.destinations)
        seen: set[str] = set()
        frontier = [START]
        while frontier:
            for nxt in targets.get(frontier.pop(), ()):
                if nxt not in seen and nxt != END:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen
