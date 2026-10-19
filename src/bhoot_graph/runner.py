"""``CompiledGraph``: runs a graph in supersteps.

Each superstep:

1. runs every scheduled task concurrently, each on a copy of the state from the step start;
2. merges their updates through the channels, in task order (so the result never depends on
   which branch finished first);
3. follows edges, routers and ``Command.goto`` from the tasks that ran to schedule the next
   step. A node triggered by several edges runs once; every ``Send`` is its own task.

The run ends when nothing is scheduled. A failed node lets its siblings finish, then its
error is raised with a note naming the node and step. Cancelling the run cancels every task.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bhoot_graph.constants import END, START
from bhoot_graph.errors import (
    GraphRecursionError,
    GraphTimeoutError,
    InvalidUpdateError,
)
from bhoot_graph.graph import Branch, NodeSpec
from bhoot_graph.state import Channel, apply_writes
from bhoot_graph.types import Command, Send

DEFAULT_RECURSION_LIMIT = 25
RunConfig = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Task:
    id: str  # "<step>:<index>:<node>", zero-padded so ids sort in scheduling order
    node: str
    arg: Any = None
    sent: bool = False  # True: ``arg`` is a Send payload, not the state


@dataclass(frozen=True, slots=True)
class StepResult:
    step: int
    tasks: tuple[Task, ...]
    updates: tuple[tuple[Task, Mapping[str, Any]], ...]
    state: Mapping[str, Any]


@dataclass(slots=True)
class _Outcome:
    update: Mapping[str, Any]
    goto: tuple[str | Send, ...] = ()


@dataclass
class CompiledGraph:
    name: str
    channels: Mapping[str, Channel]
    nodes: Mapping[str, NodeSpec]
    edges: tuple[tuple[str, str], ...]
    waiting_edges: tuple[tuple[tuple[str, ...], str], ...]
    branches: Mapping[str, tuple[Branch, ...]]
    step_timeout: float | None = None
    _edges_from: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        edges_from: dict[str, list[str]] = {}
        for a, b in self.edges:
            if b not in edges_from.setdefault(a, []):
                edges_from[a].append(b)
        self._edges_from = {k: tuple(v) for k, v in edges_from.items()}

    # ------------------------------------------------------------------ public API
    async def ainvoke(
        self, input: Mapping[str, Any] | None, config: RunConfig | None = None
    ) -> dict[str, Any]:
        """Run to the end and return the final state."""
        state: dict[str, Any] = {}
        async for result in self._steps(input, config):
            state = dict(result.state)
        return state

    def invoke(
        self, input: Mapping[str, Any] | None, config: RunConfig | None = None
    ) -> dict[str, Any]:
        """Blocking ``ainvoke`` for scripts; inside a running event loop, await ainvoke."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input, config))
        raise RuntimeError("invoke() was called inside an event loop; use await ainvoke()")

    def draw_mermaid(self) -> str:
        """The graph as a Mermaid flowchart (dotted arrows are conditional)."""
        label = {START: "__start__([start])", END: "__end__([end])"}
        lines = ["graph TD"]
        for a, b in self.edges:
            lines.append(f"    {label.get(a, a)} --> {label.get(b, b)}")
        for starts, b in self.waiting_edges:
            lines.extend(f"    {a} ==> {label.get(b, b)}" for a in starts)
        for source, branches in self.branches.items():
            for branch in branches:
                targets = sorted(set((branch.path_map or {}).values()))
                lines.extend(
                    f"    {label.get(source, source)} -.-> {label.get(t, t)}" for t in targets
                )
        for spec in self.nodes.values():
            lines.extend(f"    {spec.name} -.-> {label.get(t, t)}" for t in spec.destinations)
        return "\n".join(lines)

    # ------------------------------------------------------------------ the loop
    async def _steps(
        self, input: Mapping[str, Any] | None, config: RunConfig | None
    ) -> AsyncIterator[StepResult]:
        config = dict(config or {})
        limit = int(config.get("recursion_limit", DEFAULT_RECURSION_LIMIT))
        cap = config.get("max_concurrency")
        semaphore = asyncio.Semaphore(int(cap)) if cap else contextlib.nullcontext()
        state = apply_writes({}, self.channels, [("input", dict(input or {}))])
        yield StepResult(0, (), (), state)  # the input, so a run where nothing fires returns it
        waiting: dict[int, set[str]] = {}
        tasks = await self._schedule(0, [(START, ())], state, config, waiting)
        step = 0
        while tasks:
            step += 1
            if step > limit:
                raise GraphRecursionError(
                    f"{self.name}: hit recursion_limit={limit} supersteps without reaching END "
                    f"(next: {sorted({t.node for t in tasks})}); raise it in the run config "
                    "if the loop is expected to run longer"
                )
            outcomes = await self._run_step(step, tasks, state, config, semaphore)
            ordered = [(task, outcomes[task.id]) for task in tasks]
            state = apply_writes(
                state, self.channels, [(f"node {t.node!r}", o.update) for t, o in ordered]
            )
            yield StepResult(step, tuple(tasks), tuple((t, o.update) for t, o in ordered), state)
            finished = [(t.node, o.goto) for t, o in ordered]
            tasks = await self._schedule(step, finished, state, config, waiting)

    async def _run_step(
        self,
        step: int,
        tasks: Sequence[Task],
        state: Mapping[str, Any],
        config: RunConfig,
        semaphore: contextlib.AbstractAsyncContextManager[Any],
    ) -> dict[str, _Outcome]:
        running = {
            task.id: asyncio.create_task(self._run_task(task, state, config, semaphore))
            for task in tasks
        }
        try:
            _, pending = await asyncio.wait(running.values(), timeout=self.step_timeout)
        except BaseException:  # cancelled from outside: take every branch down with us
            await _cancel_all(running.values())
            raise
        if pending:
            await _cancel_all(pending)
            slow = sorted({t.node for t in tasks if running[t.id] in pending})
            raise GraphTimeoutError(
                f"{self.name}: step {step} exceeded step_timeout={self.step_timeout}s; "
                f"still running: {slow}"
            )
        failed = [t for t in tasks if running[t.id].exception() is not None]
        if failed:
            error = running[failed[0].id].exception()
            assert error is not None
            error.add_note(f"in node {failed[0].node!r} (task {failed[0].id}) of {self.name}")
            if len(failed) > 1:
                error.add_note(f"{len(failed) - 1} sibling task(s) failed as well")
            raise error
        return {task_id: future.result() for task_id, future in running.items()}

    async def _run_task(
        self,
        task: Task,
        state: Mapping[str, Any],
        config: RunConfig,
        semaphore: contextlib.AbstractAsyncContextManager[Any],
    ) -> _Outcome:
        spec = self.nodes[task.node]
        arg = task.arg if task.sent else dict(state)
        async with semaphore:
            if isinstance(spec.action, CompiledGraph):
                sub_input = {k: v for k, v in arg.items() if k in spec.action.channels}
                result: Any = await spec.action.ainvoke(sub_input, _child_config(config))
                result = {k: v for k, v in result.items() if k in self.channels}
            elif spec.is_async:
                result = await (
                    spec.action(arg, config=config) if spec.takes_config else spec.action(arg)
                )
            elif spec.takes_config:
                result = await asyncio.to_thread(spec.action, arg, config=config)
            else:
                result = await asyncio.to_thread(spec.action, arg)
        return self._outcome(task.node, result)

    def _outcome(self, node: str, result: Any) -> _Outcome:
        if result is None:
            return _Outcome({})
        if isinstance(result, Command):
            goto = result.goto
            targets = (goto,) if isinstance(goto, str | Send) else tuple(goto)
            for target in targets:
                self._check_target(target, f"Command(goto=...) from {node!r}")
            return _Outcome(dict(result.update or {}), targets)
        if isinstance(result, Mapping):
            return _Outcome(dict(result))
        raise InvalidUpdateError(
            f"node {node!r} returned {type(result).__name__}; expected a dict, Command or None"
        )

    # ------------------------------------------------------------------ routing
    async def _schedule(
        self,
        step: int,
        finished: Sequence[tuple[str, tuple[str | Send, ...]]],
        state: Mapping[str, Any],
        config: RunConfig,
        waiting: dict[int, set[str]],
    ) -> list[Task]:
        targets: list[str | Send] = []
        for node, goto in finished:
            targets.extend(self._edges_from.get(node, ()))
            for branch in self.branches.get(node, ()):
                targets.extend(await self._route(node, branch, state, config))
            targets.extend(goto)
        ran = {node for node, _ in finished}
        for index, (starts, join_target) in enumerate(self.waiting_edges):
            done = waiting.setdefault(index, set())
            done.update(ran & set(starts))
            if done >= set(starts):
                done.clear()
                targets.append(join_target)

        tasks: list[Task] = []
        scheduled: set[str] = set()
        for target in targets:
            if isinstance(target, Send):
                tasks.append(
                    Task(_task_id(step + 1, len(tasks), target.node), target.node, target.arg, True)
                )
            elif target != END and target not in scheduled:
                scheduled.add(target)
                tasks.append(Task(_task_id(step + 1, len(tasks), target), target))
        return tasks

    async def _route(
        self, source: str, branch: Branch, state: Mapping[str, Any], config: RunConfig
    ) -> list[str | Send]:
        args: tuple[Any, ...] = (dict(state),)
        kwargs = {"config": config} if branch.takes_config else {}
        raw = branch.path(*args, **kwargs)
        if branch.is_async or asyncio.iscoroutine(raw):
            raw = await raw
        items = raw if isinstance(raw, list | tuple) else [raw]
        where = f"router after {source!r}"
        resolved: list[str | Send] = []
        for item in items:
            if isinstance(item, Send):
                target: str | Send = item
            elif branch.path_map is not None:
                if not isinstance(item, Hashable) or item not in branch.path_map:
                    raise InvalidUpdateError(
                        f"{where} returned {item!r}, which is not in its path_map "
                        f"{sorted(map(str, branch.path_map))}"
                    )
                target = branch.path_map[item]
            elif isinstance(item, str):
                target = item
            else:
                raise InvalidUpdateError(f"{where} returned {item!r}; expected a node name or Send")
            self._check_target(target, where)
            resolved.append(target)
        return resolved

    def _check_target(self, target: str | Send, where: str) -> None:
        name = target.node if isinstance(target, Send) else target
        if name not in self.nodes and not (name == END and not isinstance(target, Send)):
            raise InvalidUpdateError(f"{where} pointed to unknown node {name!r}")


def _task_id(step: int, index: int, node: str) -> str:
    return f"{step:06d}:{index:06d}:{node}"


def _child_config(config: RunConfig) -> dict[str, Any]:
    child = dict(config)
    child.pop("recursion_limit", None)  # a subgraph gets its own budget of supersteps
    return child


async def _cancel_all(futures: Any) -> None:
    futures = list(futures)
    for future in futures:
        future.cancel()
    await asyncio.gather(*futures, return_exceptions=True)
