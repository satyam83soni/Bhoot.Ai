"""bhoot_graph checkpoints: saving, resuming after failures, versions, locks, serialization."""

from __future__ import annotations

import asyncio
import enum
import itertools
import operator
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import UUID

import pytest

from bhoot_graph import (
    END,
    START,
    BaseCheckpointSaver,
    CheckpointMismatchError,
    GraphError,
    GraphTimeoutError,
    JsonSerializer,
    MemorySaver,
    Send,
    SerializationError,
    SqliteSaver,
    StateGraph,
    ThreadBusyError,
)


class State(TypedDict):
    n: int
    log: Annotated[list[str], operator.add]


SaverFactory = Callable[[], BaseCheckpointSaver]


@pytest.fixture(params=["memory", "sqlite"])
def make_saver(request: pytest.FixtureRequest, tmp_path: Path) -> SaverFactory:
    if request.param == "memory":
        saver = MemorySaver()
        return lambda: saver
    return lambda: SqliteSaver(tmp_path / "checkpoints.db")


def _cfg(thread: str = "job-1", **extra: Any) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}, **extra}


class Flaky:
    """A node that counts its calls and fails the first ``fail_times`` of them."""

    def __init__(self, name: str, fail_times: int = 0) -> None:
        self.name, self.fail_times, self.calls = name, fail_times, 0

    async def __call__(self, state: State) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"{self.name} crashed")
        return {"n": state["n"] + 1, "log": [self.name]}


def _chain(nodes: list[Flaky], saver: BaseCheckpointSaver, version: str | None = None) -> Any:
    g = StateGraph(State)
    for node in nodes:
        g.add_node(node.name, node)
    g.add_edge(START, nodes[0].name)
    for a, b in itertools.pairwise(nodes):
        g.add_edge(a.name, b.name)
    g.add_edge(nodes[-1].name, END)
    return g.compile(saver, version=version)


# ---------------------------------------------------------------- saving and resuming
async def test_every_superstep_is_saved(make_saver: SaverFactory) -> None:
    app = _chain([Flaky("a"), Flaky("b")], make_saver())
    await app.ainvoke({"n": 0}, _cfg())
    history = await app.aget_state_history(_cfg())
    assert [s.next for s in history] == [(), ("b",), ("a",)]  # newest first
    latest = await app.aget_state(_cfg())
    assert latest is not None
    assert latest.values == {"n": 2, "log": ["a", "b"]}


@pytest.mark.parametrize("crash_at", ["a", "b", "c", "d"])
async def test_crash_at_any_node_resumes_without_redoing_finished_work(
    make_saver: SaverFactory, crash_at: str
) -> None:
    nodes = [Flaky(name, fail_times=1 if name == crash_at else 0) for name in "abcd"]
    app = _chain(nodes, make_saver())
    with pytest.raises(RuntimeError, match=f"{crash_at} crashed"):
        await app.ainvoke({"n": 0}, _cfg())
    resumed = await _chain(nodes, make_saver()).ainvoke(None, _cfg())  # "new process"
    assert resumed == {"n": 4, "log": ["a", "b", "c", "d"]}
    assert {n.name: n.calls for n in nodes} == {
        n.name: 2 if n.name == crash_at else 1 for n in nodes
    }


async def test_failed_fan_out_keeps_finished_siblings(make_saver: SaverFactory) -> None:
    class Fan(TypedDict):
        out: Annotated[list[int], operator.add]

    calls: Counter[int] = Counter()
    broken = {3}

    async def work(item: int) -> dict[str, Any]:
        calls[item] += 1
        if item in broken:
            raise RuntimeError(f"page {item} failed")
        return {"out": [item]}

    def build() -> Any:
        g = StateGraph(Fan)
        g.add_node("work", work)
        g.add_conditional_edges(START, lambda s: [Send("work", i) for i in range(5)], ["work"])
        return g.compile(make_saver())

    with pytest.raises(RuntimeError, match="page 3 failed"):
        await build().ainvoke({}, _cfg())
    broken.clear()
    result = await build().ainvoke(None, _cfg())
    assert result["out"] == [0, 1, 2, 3, 4]  # merged in Send order despite the retry
    assert calls == {0: 1, 1: 1, 2: 1, 3: 2, 4: 1}


async def test_timed_out_step_keeps_the_fast_branches(make_saver: SaverFactory) -> None:
    calls: Counter[str] = Counter()
    slow = {"hang": True}

    async def fast(state: State) -> dict[str, Any]:
        calls["fast"] += 1
        return {"log": ["fast"]}

    async def hang(state: State) -> dict[str, Any]:
        calls["hang"] += 1
        if slow["hang"]:
            await asyncio.sleep(10)
        return {"log": ["hang"]}

    def build() -> Any:
        g = StateGraph(State)
        g.add_node("fast", fast)
        g.add_node("hang", hang)
        g.add_edge(START, "fast")
        g.add_edge(START, "hang")
        app = g.compile(make_saver())
        app.step_timeout = 0.05
        return app

    with pytest.raises(GraphTimeoutError):
        await build().ainvoke({"n": 0}, _cfg())
    slow["hang"] = False
    assert (await build().ainvoke(None, _cfg()))["log"] == ["fast", "hang"]
    assert calls == {"fast": 1, "hang": 2}


async def test_waiting_edge_progress_survives_a_crash(make_saver: SaverFactory) -> None:
    long2 = Flaky("long2", fail_times=1)
    joined: list[str] = []

    def build() -> Any:
        g = StateGraph(State)
        g.add_node("short", lambda s: {"log": ["short"]})
        g.add_node("long1", lambda s: {"log": ["long1"]})
        g.add_node("long2", long2)
        g.add_node("join", lambda s: joined.append("join") or {"log": ["join"]})
        g.add_edge(START, "short")
        g.add_edge(START, "long1")
        g.add_edge("long1", "long2")
        g.add_edge(["short", "long2"], "join")
        return g.compile(make_saver())

    with pytest.raises(RuntimeError):
        await build().ainvoke({"n": 0}, _cfg())
    result = await build().ainvoke(None, _cfg())
    assert joined == ["join"]
    assert result["log"] == ["short", "long1", "long2", "join"]


async def test_cancelled_run_resumes_from_its_last_checkpoint(make_saver: SaverFactory) -> None:
    started = asyncio.Event()
    hold = {"on": True}

    async def wait(state: State) -> dict[str, Any]:
        started.set()
        if hold["on"]:
            await asyncio.sleep(10)
        return {"log": ["wait"]}

    def build() -> Any:
        g = StateGraph(State)
        g.add_node("first", lambda s: {"log": ["first"]})
        g.add_node("wait", wait)
        g.add_edge(START, "first")
        g.add_edge("first", "wait")
        return g.compile(make_saver())

    run = asyncio.create_task(build().ainvoke({"n": 0}, _cfg()))
    await started.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    hold["on"] = False
    assert (await build().ainvoke(None, _cfg()))["log"] == ["first", "wait"]


async def test_new_input_continues_a_finished_thread(make_saver: SaverFactory) -> None:
    app = _chain([Flaky("a")], make_saver())
    await app.ainvoke({"n": 0}, _cfg())
    result = await app.ainvoke({"log": ["again"]}, _cfg())
    assert result == {"n": 2, "log": ["a", "again", "a"]}


async def test_resuming_a_finished_thread_returns_its_state(make_saver: SaverFactory) -> None:
    nodes = [Flaky("a")]
    app = _chain(nodes, make_saver())
    first = await app.ainvoke({"n": 0}, _cfg())
    assert await app.ainvoke(None, _cfg()) == first
    assert nodes[0].calls == 1


async def test_threads_are_independent(make_saver: SaverFactory) -> None:
    app = _chain([Flaky("a")], make_saver())
    await app.ainvoke({"n": 10}, _cfg("one"))
    await app.ainvoke({"n": 20}, _cfg("two"))
    one, two = await app.aget_state(_cfg("one")), await app.aget_state(_cfg("two"))
    assert one is not None
    assert two is not None
    assert (one.values["n"], two.values["n"]) == (11, 21)


# ---------------------------------------------------------------- versions
async def test_thread_is_pinned_to_its_graph_version(make_saver: SaverFactory) -> None:
    await _chain([Flaky("a")], make_saver(), version="1").ainvoke({"n": 0}, _cfg())
    with pytest.raises(CheckpointMismatchError, match="version '1'"):
        await _chain([Flaky("a")], make_saver(), version="2").ainvoke(None, _cfg())


async def test_changed_graph_cannot_resume_an_unfinished_thread(make_saver: SaverFactory) -> None:
    with pytest.raises(RuntimeError):
        await _chain([Flaky("a"), Flaky("b", 1)], make_saver()).ainvoke({"n": 0}, _cfg())
    changed = _chain([Flaky("a"), Flaky("b"), Flaky("c")], make_saver())
    with pytest.raises(CheckpointMismatchError, match="bump the graph version"):
        await changed.ainvoke(None, _cfg())


# ---------------------------------------------------------------- safety
async def test_run_config_is_never_saved(tmp_path: Path) -> None:
    path = tmp_path / "secret.db"
    saver = SqliteSaver(path)
    seen: list[str] = []

    def node(state: State, config: dict[str, Any]) -> dict[str, Any]:
        seen.append(config["configurable"]["pdf_password"])
        return {"n": 1}

    g = StateGraph(State)
    g.add_node("open", node)
    g.add_edge(START, "open")
    config = {"configurable": {"thread_id": "job-1", "pdf_password": "hunter2-secret"}}
    await g.compile(saver).ainvoke({"n": 0}, config)
    saver.close()
    assert seen == ["hunter2-secret"]
    assert b"hunter2-secret" not in path.read_bytes()


async def test_two_runs_cannot_share_a_thread(make_saver: SaverFactory) -> None:
    release = asyncio.Event()
    entered = asyncio.Event()

    async def hold(state: State) -> dict[str, Any]:
        entered.set()
        await release.wait()
        return {"n": 1}

    g = StateGraph(State)
    g.add_node("hold", hold)
    g.add_edge(START, "hold")
    app = g.compile(make_saver())
    first = asyncio.create_task(app.ainvoke({"n": 0}, _cfg()))
    await entered.wait()
    with pytest.raises(ThreadBusyError):
        await app.ainvoke({"n": 0}, _cfg())
    release.set()
    await first
    assert (await app.ainvoke(None, _cfg()))["n"] == 1  # lock released afterwards


async def test_expired_lock_can_be_taken_over(make_saver: SaverFactory) -> None:
    saver = make_saver()
    assert await saver.aacquire("t", "crashed-process", ttl_s=0.0)
    assert await saver.aacquire("t", "new-process", ttl_s=60)
    assert not await saver.aacquire("t", "third", ttl_s=60)


async def test_checkpointer_needs_a_thread_id() -> None:
    app = _chain([Flaky("a")], MemorySaver())
    with pytest.raises(GraphError, match="thread_id"):
        await app.ainvoke({"n": 0})
    with pytest.raises(GraphError, match="without a checkpointer"):
        await _chain_without_saver().aget_state(_cfg())


def _chain_without_saver() -> Any:
    g = StateGraph(State)
    g.add_node("a", Flaky("a"))
    g.add_edge(START, "a")
    return g.compile()


async def test_large_state_round_trips(tmp_path: Path) -> None:
    class Big(TypedDict):
        text: str

    g = StateGraph(Big)
    g.add_node("grow", lambda s: {"text": s["text"] * 2})
    g.add_edge(START, "grow")
    saver = SqliteSaver(tmp_path / "big.db")
    await g.compile(saver).ainvoke({"text": "x" * 3_000_000}, _cfg())
    snapshot = await g.compile(saver).aget_state(_cfg())
    assert snapshot is not None
    assert len(snapshot.values["text"]) == 6_000_000


def test_sync_get_state() -> None:
    app = _chain([Flaky("a")], MemorySaver())
    app.invoke({"n": 0}, _cfg())
    snapshot = app.get_state(_cfg())
    assert snapshot is not None
    assert snapshot.values["n"] == 1


# ---------------------------------------------------------------- serializer
class Status(enum.Enum):
    OK = "ok"
    BAD = "bad"


class Level(enum.IntEnum):
    LOW = 1


@dataclass(frozen=True)
class Field:
    path: str
    amount: Decimal
    status: Status


def test_serializer_round_trips_our_types() -> None:
    serde = JsonSerializer()
    serde.register_dataclass(Field)
    serde.register_enum(Status)
    serde.register_enum(Level)
    value = {
        "amount": Decimal("1250.50"),
        "when": datetime(2026, 3, 1, 9, 30, tzinfo=UTC),
        "day": date(2026, 3, 1),
        "at": time(9, 30),
        "gap": timedelta(days=2, seconds=5),
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "blob": b"\x00\xffpng",
        "pair": (1, "a"),
        "tags": {"x"},
        "frozen": frozenset({1}),
        "by_page": {1: "first", 2: "second"},
        "looks_tagged": {"__t": "decimal", "v": "not really"},
        "send": Send("work", {"page": 3}),
        "field": Field("summary.total", Decimal("7"), Status.OK),
        "level": Level.LOW,
        "nested": [{"f": [Field("a", Decimal("0.1"), Status.BAD)]}],
    }
    restored = serde.loads(serde.dumps(value))
    assert restored == value
    assert type(restored["level"]) is Level
    assert type(restored["pair"]) is tuple


def test_serializer_errors_are_clear() -> None:
    serde = JsonSerializer()
    with pytest.raises(SerializationError, match="register it"):
        serde.dumps({"field": Field("a", Decimal(1), Status.OK)})
    other = JsonSerializer()
    other.register_dataclass(Field)
    other.register_enum(Status)
    with pytest.raises(SerializationError, match="not registered here"):
        serde.loads(other.dumps(Field("a", Decimal(1), Status.OK)))
    with pytest.raises(SerializationError, match="version 99"):
        serde.loads(b'{"format":"bhoot_graph.json","version":99,"data":null}')
    with pytest.raises(SerializationError, match="not valid JSON"):
        serde.loads(b"\x00garbage")


async def test_unserializable_state_fails_before_anything_is_stored() -> None:
    saver = MemorySaver()
    g = StateGraph(State)
    g.add_node("a", lambda s: {"log": [object()]})
    g.add_edge(START, "a")
    with pytest.raises(SerializationError):
        await g.compile(saver).ainvoke({"n": 0}, _cfg())
    history = await saver.alist("job-1")
    assert [c.step for c in history] == [0]  # only the input checkpoint was written
