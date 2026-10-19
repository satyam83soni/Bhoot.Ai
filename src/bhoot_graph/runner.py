"""``CompiledGraph``: runs a graph in supersteps.

Each superstep:

1. runs every scheduled task concurrently, each on a copy of the state from the step start;
2. merges their updates through the channels, in task order (so the result never depends on
   which branch finished first);
3. follows edges, routers and ``Command.goto`` from the tasks that ran to schedule the next
   step. A node triggered by several edges runs once; every ``Send`` is its own task;
4. with a checkpointer, saves the state and the scheduled tasks.

The run ends when nothing is scheduled. A failed node lets its siblings finish, then its
error is raised with a note naming the node and step; with a checkpointer the siblings'
results are kept, so a resumed run (``ainvoke(None, config)``) only redoes what failed.
Cancelling the run cancels every task.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bhoot_graph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMismatchError,
    PendingWrite,
    Task,
    ThreadBusyError,
)
from bhoot_graph.constants import END, START
from bhoot_graph.errors import (
    GraphError,
    GraphRecursionError,
    GraphTimeoutError,
    InvalidUpdateError,
)
from bhoot_graph.graph import Branch, NodeSpec
from bhoot_graph.interrupt import GraphInterrupt, start_task
from bhoot_graph.state import Channel, apply_writes
from bhoot_graph.stream import RunStream, parse_modes, set_writers
from bhoot_graph.types import Command, Interrupt, Send

DEFAULT_RECURSION_LIMIT = 25
DEFAULT_LOCK_TTL_S = 900.0  # refreshed every superstep
DEFAULT_STREAM_BUFFER = 256  # events waiting for the consumer before the run pauses
RunConfig = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StepResult:
    step: int
    tasks: tuple[Task, ...]
    updates: tuple[tuple[Task, Mapping[str, Any]], ...]
    state: Mapping[str, Any]
    interrupts: tuple[Interrupt, ...] = ()  # set on the last result of a paused run


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """A saved point of a thread, as returned by ``aget_state``."""

    values: Mapping[str, Any]
    next: tuple[str, ...]  # nodes that run when the thread is resumed; () when finished
    step: int
    checkpoint_id: str
    created_at: str
    interrupts: tuple[Interrupt, ...] = ()  # questions the paused run is waiting on


@dataclass(frozen=True, slots=True)
class _Failed:
    error: BaseException


@dataclass(frozen=True, slots=True)
class _Paused:
    """Not an error: some tasks of the step called interrupt()."""

    by_task: dict[str, tuple[Interrupt, ...]]


_END = object()


@dataclass(slots=True)
class _Outcome:
    update: Mapping[str, Any]
    goto: tuple[str | Send, ...] = ()


@dataclass(slots=True)
class _Saving:
    saver: BaseCheckpointSaver
    thread: str
    owner: str
    ttl_s: float
    checkpoint_id: str = ""


@dataclass
class CompiledGraph:
    name: str
    channels: Mapping[str, Channel]
    nodes: Mapping[str, NodeSpec]
    edges: tuple[tuple[str, str], ...]
    waiting_edges: tuple[tuple[tuple[str, ...], str], ...]
    branches: Mapping[str, tuple[Branch, ...]]
    checkpointer: BaseCheckpointSaver | None = None
    version: str | None = None
    step_timeout: float | None = None
    _edges_from: dict[str, tuple[str, ...]] = field(init=False, repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        edges_from: dict[str, list[str]] = {}
        for a, b in self.edges:
            if b not in edges_from.setdefault(a, []):
                edges_from[a].append(b)
        self._edges_from = {k: tuple(v) for k, v in edges_from.items()}
        shape = {
            "nodes": sorted(self.nodes),
            "edges": sorted(self.edges),
            "waiting": sorted([sorted(s), t] for s, t in self.waiting_edges),
            "branches": sorted(
                [src, sorted(set((b.path_map or {}).values()))]
                for src, bs in self.branches.items()
                for b in bs
            ),
        }
        blob = json.dumps(shape, sort_keys=True).encode("utf-8")
        self.fingerprint = hashlib.sha256(blob).hexdigest()[:16]

    # ------------------------------------------------------------------ public API
    async def ainvoke(
        self, input: Mapping[str, Any] | Command | None, config: RunConfig | None = None
    ) -> dict[str, Any]:
        """Run to the end and return the final state.

        With a checkpointer, ``input=None`` resumes the thread from its last checkpoint,
        ``Command(resume=answer)`` answers a pending ``interrupt()`` and resumes, and an input
        starts a new run on top of the thread's saved state. A paused run returns its state
        with the pending questions under ``"__interrupt__"``.
        """
        state: dict[str, Any] = {}
        async for result in self._steps(input, config):
            state = dict(result.state)
            if result.interrupts:
                state["__interrupt__"] = result.interrupts
        return state

    async def astream(
        self,
        input: Mapping[str, Any] | Command | None,
        config: RunConfig | None = None,
        *,
        stream_mode: str | Sequence[str] = "values",
    ) -> AsyncIterator[Any]:
        """Run the graph and yield its events as they happen (modes: see ``bhoot_graph.stream``).

        One mode yields bare chunks; a list of modes yields ``(mode, chunk)`` pairs. If the
        consumer stops early (``break``), the run is cancelled and, with a checkpointer, can
        be resumed from its last saved step. An error is raised after the events before it.
        """
        modes, single = parse_modes(stream_mode)
        buffer = int((config or {}).get("stream_buffer", DEFAULT_STREAM_BUFFER))
        stream = RunStream(modes, buffer)
        producer = asyncio.create_task(self._produce(input, config, stream))
        try:
            while True:
                item = await stream.queue.get()
                if item is _END:
                    return
                if isinstance(item, _Failed):
                    raise item.error
                mode, chunk = item
                yield chunk if single else (mode, chunk)
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _produce(
        self,
        input: Mapping[str, Any] | Command | None,
        config: RunConfig | None,
        stream: RunStream,
    ) -> None:
        try:
            async for result in self._steps(input, config, stream):
                if result.interrupts:
                    await stream.push("updates", {"__interrupt__": result.interrupts})
                    await stream.push(
                        "values", {**result.state, "__interrupt__": result.interrupts}
                    )
                    continue
                for task, update in result.updates:
                    await stream.push("updates", {task.node: dict(update)})
                await stream.push("values", dict(result.state))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # handed to the consumer, after the events before it
            await stream.close(_Failed(exc))
            return
        await stream.close(_END)

    def invoke(
        self, input: Mapping[str, Any] | Command | None, config: RunConfig | None = None
    ) -> dict[str, Any]:
        """Blocking ``ainvoke`` for scripts; inside a running event loop, await ainvoke."""
        _require_no_loop("invoke", "ainvoke")
        return asyncio.run(self.ainvoke(input, config))

    async def aget_state(self, config: RunConfig) -> StateSnapshot | None:
        """The latest saved point of the thread in ``config``, or None."""
        saver, thread = self._saver_and_thread(config)
        latest = await saver.aget(thread)
        if latest is None:
            return None
        writes = await saver.aget_writes(thread, latest.id)
        paused = tuple(i for w in writes.values() if w.kind == "paused" for i in w.interrupts)
        return _snapshot(latest, paused)

    def get_state(self, config: RunConfig) -> StateSnapshot | None:
        _require_no_loop("get_state", "aget_state")
        return asyncio.run(self.aget_state(config))

    async def aget_state_history(self, config: RunConfig) -> list[StateSnapshot]:
        """Every saved point of the thread, newest first."""
        saver, thread = self._saver_and_thread(config)
        return [_snapshot(c) for c in reversed(await saver.alist(thread))]

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
        self,
        input: Mapping[str, Any] | Command | None,
        config: RunConfig | None,
        stream: RunStream | None = None,
    ) -> AsyncIterator[StepResult]:
        config = dict(config or {})
        if self.checkpointer is None:
            async for result in self._loop(input, config, None, stream):
                yield result
            return
        saver, thread = self._saver_and_thread(config)
        saving = _Saving(
            saver, thread, uuid.uuid4().hex, float(config.get("lock_ttl", DEFAULT_LOCK_TTL_S))
        )
        if not await saver.aacquire(thread, saving.owner, saving.ttl_s):
            raise ThreadBusyError(f"thread {thread!r} is being run by another process")
        try:
            async for result in self._loop(input, config, saving, stream):
                yield result
        finally:
            await saver.arelease(thread, saving.owner)

    async def _loop(
        self,
        input: Mapping[str, Any] | Command | None,
        config: RunConfig,
        saving: _Saving | None,
        stream: RunStream | None,
    ) -> AsyncIterator[StepResult]:
        limit = int(config.get("recursion_limit", DEFAULT_RECURSION_LIMIT))
        cap = config.get("max_concurrency")
        semaphore = asyncio.Semaphore(int(cap)) if cap else contextlib.nullcontext()
        latest = await saving.saver.aget(saving.thread) if saving else None
        if latest is not None:
            self._check_compatible(latest)
        done: dict[str, _Outcome] = {}
        resumes: dict[str, tuple[Any, ...]] = {}
        if isinstance(input, Command):
            if input.update is not None or input.goto or not input.has_resume:
                raise GraphError("as run input, a Command can only carry resume=...")
            if saving is None or latest is None:
                raise GraphError("Command(resume=...) needs a checkpointer and a paused thread")
        if saving and latest is not None and (input is None or isinstance(input, Command)):
            state, step = dict(latest.values), latest.step
            waiting = {index: set(sources) for index, sources in latest.waiting.items()}
            tasks = list(latest.next)
            saving.checkpoint_id = latest.id
            writes = await saving.saver.aget_writes(saving.thread, latest.id)
            if isinstance(input, Command):
                writes = await self._answer(saving, writes, input.resume)
            done = {t: _Outcome(w.update, w.goto) for t, w in writes.items() if w.kind == "done"}
            resumes = {t: w.resumes for t, w in writes.items() if w.kind == "paused"}
        else:
            assert not isinstance(input, Command)  # a Command input either resumed or raised
            base = dict(latest.values) if latest else {}
            step = latest.step + 1 if latest else 0
            state = apply_writes(base, self.channels, [("input", dict(input or {}))])
            waiting = {}
            tasks = await self._schedule(step, [(START, ())], state, config, waiting)
            await self._save(saving, step, state, tasks, waiting, stream)
        yield StepResult(
            step, (), (), state
        )  # the starting state: a run where nothing fires returns it

        executed = 0
        while tasks:
            executed += 1
            if executed > limit:
                raise GraphRecursionError(
                    f"{self.name}: hit recursion_limit={limit} supersteps without reaching END "
                    f"(next: {sorted({t.node for t in tasks})}); raise it in the run config "
                    "if the loop is expected to run longer"
                )
            step += 1
            outcomes, failure = await self._run_step(
                step, tasks, state, config, semaphore, done, resumes, stream
            )
            if isinstance(failure, _Paused):
                if saving is None:
                    raise GraphError(
                        "interrupt() needs a checkpointer: compile(checkpointer=...) and pass "
                        "a thread_id, so the run can be resumed later"
                    )
                for task in tasks:
                    if task.id in outcomes and task.id not in done:
                        o = outcomes[task.id]
                        write = PendingWrite(task.id, o.update, o.goto)
                        await saving.saver.aput_write(saving.thread, saving.checkpoint_id, write)
                    elif task.id in failure.by_task:
                        write = PendingWrite(
                            task.id,
                            kind="paused",
                            interrupts=failure.by_task[task.id],
                            resumes=resumes.get(task.id, ()),
                        )
                        await saving.saver.aput_write(saving.thread, saving.checkpoint_id, write)
                pending = tuple(i for t in tasks for i in failure.by_task.get(t.id, ()))
                if stream is not None:
                    event = {"type": "interrupt", "step": step, "interrupts": pending}
                    await stream.push("debug", event)
                yield StepResult(step - 1, (), (), state, pending)
                return
            if failure is not None:
                if saving is not None:
                    for task in tasks:
                        if task.id in outcomes and task.id not in done:
                            o = outcomes[task.id]
                            write = PendingWrite(task.id, o.update, o.goto)
                            await saving.saver.aput_write(
                                saving.thread, saving.checkpoint_id, write
                            )
                raise failure
            ordered = [(task, outcomes[task.id]) for task in tasks]
            state = apply_writes(
                state, self.channels, [(f"node {t.node!r}", o.update) for t, o in ordered]
            )
            finished = [(t.node, o.goto) for t, o in ordered]
            tasks = await self._schedule(step, finished, state, config, waiting)
            done, resumes = {}, {}
            await self._save(saving, step, state, tasks, waiting, stream)
            yield StepResult(
                step, tuple(t for t, _ in ordered), tuple((t, o.update) for t, o in ordered), state
            )

    async def _save(
        self,
        saving: _Saving | None,
        step: int,
        state: Mapping[str, Any],
        tasks: Sequence[Task],
        waiting: Mapping[int, set[str]],
        stream: RunStream | None,
    ) -> None:
        if saving is None:
            return
        if not await saving.saver.aacquire(saving.thread, saving.owner, saving.ttl_s):
            raise ThreadBusyError(
                f"lost the lock on thread {saving.thread!r} (held longer than lock_ttl?)"
            )
        checkpoint = Checkpoint(
            thread_id=saving.thread,
            step=step,
            values=dict(state),
            next=tuple(tasks),
            waiting={index: tuple(sorted(s)) for index, s in waiting.items() if s},
            graph=self.name,
            version=self.version,
            fingerprint=self.fingerprint,
        )
        await saving.saver.aput(checkpoint)
        saving.checkpoint_id = checkpoint.id
        if stream is not None:
            event = {
                "type": "checkpoint",
                "step": step,
                "checkpoint_id": checkpoint.id,
                "next": [t.node for t in tasks],
            }
            await stream.push("debug", event)

    async def _answer(
        self, saving: _Saving, writes: dict[str, PendingWrite], resume: Any
    ) -> dict[str, PendingWrite]:
        """Attach ``resume`` to the paused task(s) it answers and save that before running,
        so an answer is never lost to a crash."""
        pending = {
            i.id: task_id
            for task_id, w in writes.items()
            if w.kind == "paused"
            for i in w.interrupts
        }
        if not pending:
            raise GraphError(f"thread {saving.thread!r} has no pending interrupt to answer")
        if isinstance(resume, Mapping) and any(key in pending for key in resume):
            unknown = sorted(str(key) for key in resume if key not in pending)
            if unknown:
                raise GraphError(
                    f"no pending interrupt with id {unknown}; pending: {sorted(pending)}"
                )
            answers = dict(resume)
        elif len(pending) == 1:
            answers = {next(iter(pending)): resume}
        else:
            raise GraphError(
                f"{len(pending)} interrupts are pending; resume with "
                f"Command(resume={{interrupt_id: answer}}) using ids {sorted(pending)}"
            )
        updated = dict(writes)
        for interrupt_id, answer in answers.items():
            old = updated[pending[interrupt_id]]
            new = PendingWrite(old.task_id, kind="paused", resumes=(*old.resumes, answer))
            await saving.saver.aput_write(saving.thread, saving.checkpoint_id, new)
            updated[old.task_id] = new
        return updated

    def _check_compatible(self, saved: Checkpoint) -> None:
        if (saved.graph, saved.version) != (self.name, self.version):
            raise CheckpointMismatchError(
                f"thread {saved.thread_id!r} was saved by graph {saved.graph!r} version "
                f"{saved.version!r}; this is {self.name!r} version {self.version!r}"
            )
        if saved.fingerprint != self.fingerprint and not saved.done:
            raise CheckpointMismatchError(
                f"thread {saved.thread_id!r} stopped mid-run in a graph whose nodes or edges "
                "have since changed; finish it with the old graph, or bump the graph version "
                "and start a new thread"
            )

    def _saver_and_thread(self, config: RunConfig | None) -> tuple[BaseCheckpointSaver, str]:
        if self.checkpointer is None:
            raise GraphError(f"{self.name} was compiled without a checkpointer")
        thread = (config or {}).get("configurable", {}).get("thread_id")
        if not thread:
            raise GraphError(
                "a graph with a checkpointer needs config={'configurable': {'thread_id': ...}}"
            )
        return self.checkpointer, str(thread)

    async def _run_step(
        self,
        step: int,
        tasks: Sequence[Task],
        state: Mapping[str, Any],
        config: RunConfig,
        semaphore: contextlib.AbstractAsyncContextManager[Any],
        done: Mapping[str, _Outcome],
        resumes: Mapping[str, tuple[Any, ...]],
        stream: RunStream | None,
    ) -> tuple[dict[str, _Outcome], BaseException | _Paused | None]:
        """Run the step's tasks (reusing ``done`` ones). Returns the outcomes of every task
        that finished, and the error to raise if the step as a whole failed."""
        outcomes = {task_id: done[task_id] for task_id in (t.id for t in tasks) if task_id in done}
        running = {
            task.id: asyncio.create_task(
                self._run_task(task, state, config, semaphore, resumes.get(task.id, ()), stream)
            )
            for task in tasks
            if task.id not in done
        }
        if not running:
            return outcomes, None
        try:
            _, pending = await asyncio.wait(running.values(), timeout=self.step_timeout)
        except BaseException:  # cancelled from outside: take every branch down with us
            await _cancel_all(running.values())
            raise
        if pending:
            await _cancel_all(pending)
        failed: list[Task] = []
        paused: dict[str, tuple[Interrupt, ...]] = {}
        for task in tasks:
            future = running.get(task.id)
            if future is None or future in pending:
                continue
            error = future.exception()
            if error is None:
                outcomes[task.id] = future.result()
            elif isinstance(error, GraphInterrupt):
                paused[task.id] = (error.interrupt,)
            else:
                failed.append(task)
        if pending:
            slow = sorted({t.node for t in tasks if running.get(t.id) in pending})
            return outcomes, GraphTimeoutError(
                f"{self.name}: step {step} exceeded step_timeout={self.step_timeout}s; "
                f"still running: {slow}"
            )
        if failed:
            error = running[failed[0].id].exception()
            assert error is not None
            error.add_note(f"in node {failed[0].node!r} (task {failed[0].id}) of {self.name}")
            if len(failed) > 1:
                error.add_note(f"{len(failed) - 1} sibling task(s) failed as well")
            return outcomes, error
        if paused:
            return outcomes, _Paused(paused)
        return outcomes, None

    async def _run_task(
        self,
        task: Task,
        state: Mapping[str, Any],
        config: RunConfig,
        semaphore: contextlib.AbstractAsyncContextManager[Any],
        resumes: tuple[Any, ...],
        stream: RunStream | None,
    ) -> _Outcome:
        start_task(task.id, resumes)
        if stream is None:
            return await self._call_node(task, state, config, semaphore)
        set_writers(stream.writers_for(task.node, task.id))
        step = int(task.id.split(":", 1)[0])
        stream.task_event({"type": "task", "step": step, "task_id": task.id, "node": task.node})
        started = time.perf_counter()
        error: str | None = None
        try:
            return await self._call_node(task, state, config, semaphore)
        except GraphInterrupt:
            error = "interrupted"
            raise
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stream.task_event(
                {
                    "type": "task_result",
                    "step": step,
                    "task_id": task.id,
                    "node": task.node,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": error,
                }
            )

    async def _call_node(
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
                child = _child_config(config, task.id)
                result: Any = await spec.action.ainvoke(sub_input, child)
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


def _child_config(config: RunConfig, task_id: str) -> dict[str, Any]:
    child = dict(config)
    child.pop("recursion_limit", None)  # a subgraph gets its own budget of supersteps
    configurable = dict(child.get("configurable", {}))
    if "thread_id" in configurable:  # a subgraph with its own saver gets its own thread
        configurable["thread_id"] = f"{configurable['thread_id']}/{task_id}"
        child["configurable"] = configurable
    return child


def _snapshot(checkpoint: Checkpoint, interrupts: tuple[Interrupt, ...] = ()) -> StateSnapshot:
    return StateSnapshot(
        values=dict(checkpoint.values),
        next=tuple(t.node for t in checkpoint.next),
        step=checkpoint.step,
        checkpoint_id=checkpoint.id,
        created_at=checkpoint.created_at,
        interrupts=interrupts,
    )


def _require_no_loop(name: str, async_name: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(f"{name}() was called inside an event loop; use await {async_name}()")


async def _cancel_all(futures: Any) -> None:
    futures = list(futures)
    for future in futures:
        future.cancel()
    await asyncio.gather(*futures, return_exceptions=True)
