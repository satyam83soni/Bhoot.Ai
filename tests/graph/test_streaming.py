"""bhoot_graph streaming: modes, node writers, ordering, back-pressure, early stop, errors."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import operator
import random
from typing import Annotated, Any, TypedDict

import pytest

from bhoot_graph import (
    START,
    GraphError,
    MemorySaver,
    Send,
    StateGraph,
    get_stream_writer,
)


class State(TypedDict):
    n: int
    log: Annotated[list[str], operator.add]


def _chain(*names: str) -> StateGraph:
    g = StateGraph(State)
    for name in names:
        g.add_node(name, _adder(name))
    g.add_edge(START, names[0])
    for a, b in itertools.pairwise(names):
        g.add_edge(a, b)
    return g


def _adder(name: str) -> Any:
    async def node(state: State) -> dict[str, Any]:
        return {"n": state["n"] + 1, "log": [name]}

    return node


async def _collect(stream: Any, into: list[Any] | None = None) -> list[Any]:
    seen = [] if into is None else into
    async for item in stream:
        seen.append(item)
    return seen


# ---------------------------------------------------------------- modes
async def test_values_mode_yields_the_input_then_every_step() -> None:
    app = _chain("a", "b").compile()
    values = await _collect(app.astream({"n": 0}))
    assert values == [
        {"n": 0},
        {"n": 1, "log": ["a"]},
        {"n": 2, "log": ["a", "b"]},
    ]


async def test_updates_mode_yields_one_chunk_per_task_in_task_order() -> None:
    class Fan(TypedDict):
        out: Annotated[list[int], operator.add]

    async def work(item: int) -> dict[str, Any]:
        await asyncio.sleep(random.random() / 200)
        return {"out": [item]}

    g = StateGraph(Fan)
    g.add_node("work", work)
    g.add_conditional_edges(START, lambda s: [Send("work", i) for i in range(6)], ["work"])
    app = g.compile()
    first = await _collect(app.astream({}, stream_mode="updates"))
    second = await _collect(app.astream({}, stream_mode="updates"))
    assert first == [{"work": {"out": [i]}} for i in range(6)]
    assert second == first  # same order however the branches finished


async def test_several_modes_yield_mode_chunk_pairs() -> None:
    app = _chain("a").compile()
    events = await _collect(app.astream({"n": 0}, stream_mode=["updates", "values"]))
    assert events == [
        ("values", {"n": 0}),
        ("updates", {"a": {"n": 1, "log": ["a"]}}),
        ("values", {"n": 1, "log": ["a"]}),
    ]


async def test_custom_writes_from_async_and_sync_nodes_arrive_before_the_step_update() -> None:
    async def ocr(state: State) -> dict[str, Any]:
        write = get_stream_writer()
        for page in (1, 2):
            write({"ocr_page": page})
        return {"log": ["ocr"]}

    def parse(state: State) -> dict[str, Any]:  # runs in a worker thread
        get_stream_writer()({"parsed": True})
        return {"log": ["parse"]}

    g = StateGraph(State)
    g.add_node("ocr", ocr)
    g.add_node("parse", parse)
    g.add_edge(START, "ocr")
    g.add_edge("ocr", "parse")
    events = await _collect(g.compile().astream({"n": 0}, stream_mode=["custom", "updates"]))
    assert events == [
        ("custom", {"ocr_page": 1}),
        ("custom", {"ocr_page": 2}),
        ("updates", {"ocr": {"log": ["ocr"]}}),
        ("custom", {"parsed": True}),
        ("updates", {"parse": {"log": ["parse"]}}),
    ]


async def test_messages_mode_tags_chunks_with_their_node() -> None:
    async def llm(state: State) -> dict[str, Any]:
        tokens = get_stream_writer("messages")
        for token in ("{", '"n"', "}"):
            tokens(token)
        return {}

    g = StateGraph(State)
    g.add_node("llm", llm)
    g.add_edge(START, "llm")
    chunks = await _collect(g.compile().astream({"n": 0}, stream_mode="messages"))
    assert [c for c, _ in chunks] == ["{", '"n"', "}"]
    assert all(meta["node"] == "llm" and meta["task_id"].endswith(":llm") for _, meta in chunks)


async def test_task_and_debug_events_carry_timings_errors_and_checkpoints() -> None:
    app = _chain("a", "b").compile(MemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    tasks = await _collect(
        app.astream({"n": 0}, {"configurable": {"thread_id": "u"}}, stream_mode="tasks")
    )
    assert [(e["type"], e["node"]) for e in tasks] == [
        ("task", "a"),
        ("task_result", "a"),
        ("task", "b"),
        ("task_result", "b"),
    ]
    assert all(
        e["error"] is None and e["duration_ms"] >= 0 for e in tasks if e["type"] == "task_result"
    )
    debug = await _collect(app.astream({"n": 0}, config, stream_mode="debug"))
    checkpoints = [e for e in debug if e["type"] == "checkpoint"]
    assert [c["next"] for c in checkpoints] == [["a"], ["b"], []]


async def test_failed_task_shows_up_in_task_events() -> None:
    async def boom(state: State) -> dict[str, Any]:
        raise ValueError("bad")

    g = StateGraph(State)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    seen: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="bad"):
        await _collect(g.compile().astream({"n": 0}, stream_mode="tasks"), seen)
    assert seen[-1]["error"] == "ValueError: bad"


async def test_writer_outside_astream_is_a_no_op() -> None:
    async def node(state: State) -> dict[str, Any]:
        get_stream_writer()({"ignored": True})
        return {"n": 1}

    g = StateGraph(State)
    g.add_node("node", node)
    g.add_edge(START, "node")
    assert (await g.compile().ainvoke({"n": 0}))["n"] == 1
    get_stream_writer()("outside any graph")  # also fine


async def test_subgraph_custom_writes_reach_the_parent_stream() -> None:
    class Inner(TypedDict):
        n: int

    async def inner_node(state: Inner) -> dict[str, Any]:
        get_stream_writer()("from inside")
        return {"n": state["n"] + 1}

    inner = StateGraph(Inner)
    inner.add_node("inner_node", inner_node)
    inner.add_edge(START, "inner_node")
    outer = StateGraph(Inner)
    outer.add_node("sub", inner.compile())
    outer.add_edge(START, "sub")
    assert await _collect(outer.compile().astream({"n": 0}, stream_mode="custom")) == [
        "from inside"
    ]


async def test_bad_modes_are_rejected() -> None:
    app = _chain("a").compile()
    with pytest.raises(GraphError, match="unknown stream mode"):
        await _collect(app.astream({"n": 0}, stream_mode="tokens"))
    with pytest.raises(GraphError, match="not 'values'"):
        get_stream_writer("values")


# ---------------------------------------------------------------- flow control
async def test_slow_consumer_pauses_the_run() -> None:
    executed = 0

    async def step(state: State) -> dict[str, Any]:
        nonlocal executed
        executed += 1
        return {"n": state["n"] + 1}

    g = StateGraph(State)
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_conditional_edges(
        "step", lambda s: "step" if s["n"] < 20 else "__end__", ["step", "__end__"]
    )
    app = g.compile()
    consumed = 0
    lead = 0
    async for _ in app.astream({"n": 0}, {"stream_buffer": 1, "recursion_limit": 50}):
        consumed += 1
        await asyncio.sleep(0.005)
        lead = max(lead, executed - consumed)
    assert executed == 20
    assert lead <= 2  # the graph never ran more than a step or two ahead of the reader


async def test_breaking_out_cancels_the_run_and_it_can_resume() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    hold = {"on": True}

    async def wait(state: State) -> dict[str, Any]:
        started.set()
        try:
            if hold["on"]:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"log": ["wait"]}

    saver = MemorySaver()
    g = StateGraph(State)
    g.add_node("first", lambda s: {"log": ["first"]})
    g.add_node("wait", wait)
    g.add_edge(START, "first")
    g.add_edge("first", "wait")
    app = g.compile(saver)
    config = {"configurable": {"thread_id": "job"}}
    async with contextlib.aclosing(app.astream({"n": 0}, config, stream_mode="updates")) as events:
        async for update in events:
            if "first" in update:
                await started.wait()
                break
    assert cancelled.is_set()
    hold["on"] = False
    assert (await app.ainvoke(None, config))["log"] == ["first", "wait"]


async def test_plain_break_also_stops_the_run() -> None:
    cancelled = asyncio.Event()

    async def wait(state: State) -> dict[str, Any]:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {}

    g = StateGraph(State)
    g.add_node("wait", wait)
    g.add_edge(START, "wait")
    async for _ in g.compile().astream({"n": 0}, stream_mode="tasks"):
        break
    await asyncio.wait_for(cancelled.wait(), timeout=2)


async def test_error_arrives_after_the_events_before_it() -> None:
    async def boom(state: State) -> dict[str, Any]:
        raise RuntimeError("page 3 unreadable")

    g = _chain("a", "b")
    g.add_node("boom", boom)
    g.add_edge("b", "boom")
    seen: list[Any] = []
    with pytest.raises(RuntimeError, match="page 3 unreadable"):
        await _collect(g.compile().astream({"n": 0}, stream_mode="updates"), seen)
    assert seen == [{"a": {"n": 1, "log": ["a"]}}, {"b": {"n": 2, "log": ["b"]}}]
