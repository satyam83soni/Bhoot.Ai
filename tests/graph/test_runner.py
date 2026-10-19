"""bhoot_graph core: building, routing, fan-out, merging, limits and failures."""

from __future__ import annotations

import asyncio
import operator
import random
import threading
from typing import Annotated, Any, NotRequired, TypedDict

import pytest

from bhoot_graph import (
    END,
    START,
    Command,
    GraphRecursionError,
    GraphTimeoutError,
    InvalidGraphError,
    InvalidUpdateError,
    Send,
    StateGraph,
)


class State(TypedDict):
    n: int
    log: Annotated[list[str], operator.add]
    note: NotRequired[str]


def _step(name: str, delta: int = 1) -> Any:
    async def node(state: State) -> dict[str, Any]:
        return {"n": state["n"] + delta, "log": [name]}

    node.__name__ = name
    return node


# ---------------------------------------------------------------- happy paths
async def test_linear_graph_runs_in_order() -> None:
    g = StateGraph(State)
    g.add_sequence([_step("a"), _step("b"), _step("c")])
    g.add_edge(START, "a")
    g.add_edge("c", END)
    result = await g.compile().ainvoke({"n": 0})
    assert result == {"n": 3, "log": ["a", "b", "c"]}


async def test_repair_loop_with_conditional_edges() -> None:
    g = StateGraph(State)
    g.add_node("extract", _step("extract"))
    g.add_node("repair", _step("repair"))
    g.add_edge(START, "extract")
    g.add_conditional_edges(
        "extract", lambda s: "done" if s["n"] >= 3 else "again", {"again": "repair", "done": END}
    )
    g.add_edge("repair", "extract")
    result = await g.compile().ainvoke({"n": 0})
    assert result["log"] == ["extract", "repair", "extract"]


async def test_send_fans_out_and_reducer_collects_in_send_order() -> None:
    class Fan(TypedDict):
        items: list[int]
        out: Annotated[list[int], operator.add]

    async def work(item: int) -> dict[str, Any]:
        await asyncio.sleep(random.random() / 100)  # finish in random order
        return {"out": [item * 10]}

    g = StateGraph(Fan)
    g.add_node("work", work)
    g.add_conditional_edges(START, lambda s: [Send("work", i) for i in s["items"]], ["work"])
    g.add_edge("work", END)
    result = await g.compile().ainvoke({"items": list(range(20))})
    assert result["out"] == [i * 10 for i in range(20)]


async def test_fan_out_over_zero_items_just_continues() -> None:
    class Fan(TypedDict):
        items: list[int]
        out: Annotated[list[int], operator.add]

    def route(state: Fan) -> list[Send | str]:
        return [Send("work", i) for i in state["items"]] or ["merge"]

    g = StateGraph(Fan)
    g.add_node("work", lambda i: {"out": [i]})
    g.add_node("merge", lambda s: {"out": [-1]})
    g.add_conditional_edges(START, route, ["work", "merge"])
    g.add_edge("work", "merge")
    g.add_edge("merge", END)
    assert (await g.compile().ainvoke({"items": []}))["out"] == [-1]
    assert (await g.compile().ainvoke({"items": [1, 2]}))["out"] == [1, 2, -1]


async def test_router_returning_empty_list_ends_the_run() -> None:
    g = StateGraph(State)
    g.add_node("a", _step("a"))
    g.add_conditional_edges(START, lambda s: [], ["a"])
    assert await g.compile().ainvoke({"n": 0}) == {"n": 0}


async def test_waiting_edge_joins_parallel_branches_of_different_length() -> None:
    g = StateGraph(State)
    g.add_node("short", lambda s: {"log": ["short"]})
    g.add_node("long1", lambda s: {"log": ["long1"]})
    g.add_node("long2", lambda s: {"log": ["long2"]})
    g.add_node("join", lambda s: {"log": ["join"]})
    g.add_edge(START, "short")
    g.add_edge(START, "long1")
    g.add_edge("long1", "long2")
    g.add_edge(["short", "long2"], "join")
    g.add_edge("join", END)
    result = await g.compile().ainvoke({"n": 0})
    assert result["log"] == ["short", "long1", "long2", "join"]  # join ran once, at the end


async def test_node_triggered_twice_in_a_step_runs_once() -> None:
    calls: list[str] = []
    g = StateGraph(State)
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_node("c", lambda s: calls.append("c") or {"log": ["c"]})
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    await g.compile().ainvoke({"n": 0})
    assert calls == ["c"]


async def test_command_updates_and_jumps() -> None:
    def decide(state: State) -> Command:
        return Command(update={"note": "escalated"}, goto="strong")

    g = StateGraph(State)
    g.add_node("decide", decide, destinations=["strong", END])
    g.add_node("strong", _step("strong", 10))
    g.add_edge(START, "decide")
    g.add_edge("strong", END)
    result = await g.compile().ainvoke({"n": 0})
    assert result == {"n": 10, "log": ["strong"], "note": "escalated"}


async def test_command_goto_can_fan_out_with_send() -> None:
    def plan(state: State) -> Command:
        return Command(goto=[Send("work", k) for k in ("x", "y")])

    g = StateGraph(State)
    g.add_node("plan", plan, destinations=["work"])
    g.add_node("work", lambda key: {"log": [key]})
    g.add_edge(START, "plan")
    assert (await g.compile().ainvoke({"n": 0}))["log"] == ["x", "y"]


async def test_sync_nodes_run_off_the_event_loop_and_get_config() -> None:
    seen: dict[str, Any] = {}

    def blocking(state: State, config: dict[str, Any]) -> dict[str, Any]:
        seen["thread"] = threading.current_thread() is not threading.main_thread()
        seen["job"] = config["configurable"]["thread_id"]
        return {"n": 1}

    g = StateGraph(State)
    g.add_node("blocking", blocking)
    g.add_edge(START, "blocking")
    await g.compile().ainvoke({"n": 0}, {"configurable": {"thread_id": "job-1"}})
    assert seen == {"thread": True, "job": "job-1"}


async def test_async_router_and_callable_class_node() -> None:
    class Add:
        async def __call__(self, state: State) -> dict[str, Any]:
            return {"n": state["n"] + 5}

    async def route(state: State) -> str:
        return END if state["n"] > 0 else "add"

    g = StateGraph(State)
    g.add_node("add", Add())
    g.add_conditional_edges(START, route, ["add", END])
    g.add_edge("add", END)
    assert (await g.compile().ainvoke({"n": 0}))["n"] == 5


async def test_subgraph_as_a_node_shares_keys() -> None:
    class Inner(TypedDict):
        n: int
        scratch: str

    inner = StateGraph(Inner)
    inner.add_node("double", lambda s: {"n": s["n"] * 2, "scratch": "x"})
    inner.add_edge(START, "double")

    class Outer(TypedDict):
        n: int

    outer = StateGraph(Outer)
    outer.add_node("inner", inner.compile())
    outer.add_edge(START, "inner")
    assert await outer.compile().ainvoke({"n": 4}) == {"n": 8}


def test_invoke_blocks_outside_an_event_loop() -> None:
    g = StateGraph(State)
    g.add_node("a", _step("a"))
    g.add_edge(START, "a")
    assert g.compile().invoke({"n": 1})["n"] == 2


async def test_invoke_inside_an_event_loop_says_to_await() -> None:
    g = StateGraph(State)
    g.add_node("a", _step("a"))
    g.add_edge(START, "a")
    with pytest.raises(RuntimeError, match="ainvoke"):
        g.compile().invoke({"n": 1})


async def test_empty_input_is_allowed() -> None:
    g = StateGraph(State)
    g.add_node("a", lambda s: {"log": ["ran"]})
    g.add_edge(START, "a")
    assert await g.compile().ainvoke(None) == {"log": ["ran"]}


def append_in_place(old: list[int], new: list[int]) -> list[int]:
    old.extend(new)
    return old


class InPlace(TypedDict):
    xs: Annotated[list[int], append_in_place]


async def test_reducer_start_value_is_fresh_for_every_run() -> None:
    g = StateGraph(InPlace)
    g.add_node("a", lambda s: {"xs": [1]})
    g.add_edge(START, "a")
    app = g.compile()
    assert (await app.ainvoke({}))["xs"] == [1]
    assert (await app.ainvoke({}))["xs"] == [1]


# ---------------------------------------------------------------- bad updates
async def test_two_branches_writing_a_plain_key_is_an_error() -> None:
    g = StateGraph(State)
    g.add_node("a", lambda s: {"n": 1})
    g.add_node("b", lambda s: {"n": 2})
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    with pytest.raises(InvalidUpdateError, match="give it a reducer"):
        await g.compile().ainvoke({"n": 0})


async def test_unknown_key_and_wrong_return_type_are_errors() -> None:
    g = StateGraph(State)
    g.add_node("a", lambda s: {"typo": 1})
    g.add_edge(START, "a")
    with pytest.raises(InvalidUpdateError, match="unknown key 'typo'"):
        await g.compile().ainvoke({"n": 0})

    g = StateGraph(State)
    g.add_node("a", lambda s: 42)
    g.add_edge(START, "a")
    with pytest.raises(InvalidUpdateError, match="returned int"):
        await g.compile().ainvoke({"n": 0})


async def test_router_value_outside_path_map_and_unknown_goto() -> None:
    g = StateGraph(State)
    g.add_node("a", _step("a"))
    g.add_conditional_edges(START, lambda s: "nope", {"yes": "a"})
    with pytest.raises(InvalidUpdateError, match="not in its path_map"):
        await g.compile().ainvoke({"n": 0})

    g = StateGraph(State)
    g.add_node("a", lambda s: Command(goto="ghost"))
    g.add_edge(START, "a")
    with pytest.raises(InvalidUpdateError, match="unknown node 'ghost'"):
        await g.compile().ainvoke({"n": 0})


# ---------------------------------------------------------------- compile checks
def test_compile_rejects_bad_wiring() -> None:
    def graph() -> StateGraph:
        g = StateGraph(State)
        g.add_node("a", _step("a"))
        return g

    with pytest.raises(InvalidGraphError, match="no entry point"):
        graph().compile()
    with pytest.raises(InvalidGraphError, match="unknown node 'b'"):
        graph().add_edge(START, "a").add_edge("a", "b").compile()
    with pytest.raises(InvalidGraphError, match="unknown node 'zzz'"):
        graph().add_edge(START, "a").add_conditional_edges("a", lambda s: END, ["zzz"]).compile()
    with pytest.raises(InvalidGraphError, match="never reached"):
        graph().add_node("orphan", _step("o")).add_edge(START, "a").compile()
    with pytest.raises(InvalidGraphError, match="already exists"):
        graph().add_node("a", _step("a"))
    with pytest.raises(InvalidGraphError, match="same name as a state key"):
        graph().add_node("n", _step("n"))
    with pytest.raises(InvalidGraphError, match="invalid node name"):
        graph().add_node(END, _step("x"))
    with pytest.raises(InvalidGraphError, match="START cannot"):
        graph().add_edge("a", START)


def test_router_without_path_map_skips_the_reachability_check() -> None:
    g = StateGraph(State)
    g.add_node("a", _step("a"))
    g.add_node("b", _step("b"))
    g.add_conditional_edges(START, lambda s: "b")
    g.compile()  # "b" may be reached through the router, so no error


def test_compiled_graph_is_a_snapshot() -> None:
    g = StateGraph(State)
    g.add_node("a", _step("a"))
    g.add_edge(START, "a")
    app = g.compile()
    g.add_node("b", _step("b"))
    assert "b" not in app.nodes


def test_mermaid_drawing() -> None:
    g = StateGraph(State)
    g.add_node("a", _step("a"))
    g.add_edge(START, "a")
    g.add_conditional_edges("a", lambda s: END, ["a", END])
    drawing = g.compile().draw_mermaid()
    assert "__start__([start]) --> a" in drawing
    assert "a -.-> __end__([end])" in drawing


# ---------------------------------------------------------------- limits and failures
async def test_endless_loop_hits_the_recursion_limit() -> None:
    g = StateGraph(State)
    g.add_node("spin", _step("spin"))
    g.add_edge(START, "spin")
    g.add_edge("spin", "spin")
    with pytest.raises(GraphRecursionError, match="recursion_limit=5"):
        await g.compile().ainvoke({"n": 0}, {"recursion_limit": 5})


async def test_failed_branch_lets_siblings_finish_and_names_the_node() -> None:
    finished: list[str] = []

    async def slow(state: State) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        finished.append("slow")
        return {"log": ["slow"]}

    async def boom(state: State) -> dict[str, Any]:
        raise ValueError("bad page")

    g = StateGraph(State)
    g.add_node("slow", slow)
    g.add_node("boom", boom)
    g.add_edge(START, "slow")
    g.add_edge(START, "boom")
    with pytest.raises(ValueError, match="bad page") as info:
        await g.compile().ainvoke({"n": 0})
    assert finished == ["slow"]
    assert any("in node 'boom'" in note for note in info.value.__notes__)


async def test_step_timeout_cancels_the_slow_branch() -> None:
    cancelled = asyncio.Event()

    async def hang(state: State) -> dict[str, Any]:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {}

    g = StateGraph(State)
    g.add_node("hang", hang)
    g.add_edge(START, "hang")
    app = g.compile()
    app.step_timeout = 0.05
    with pytest.raises(GraphTimeoutError, match=r"still running: \['hang'\]"):
        await app.ainvoke({"n": 0})
    assert cancelled.is_set()


async def test_cancelling_the_run_cancels_every_branch() -> None:
    started = 0
    cancelled = 0
    all_started = asyncio.Event()

    async def wait(item: int) -> dict[str, Any]:
        nonlocal started, cancelled
        started += 1
        if started == 5:
            all_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled += 1
            raise
        return {}

    g = StateGraph(State)
    g.add_node("wait", wait)
    g.add_conditional_edges(START, lambda s: [Send("wait", i) for i in range(5)], ["wait"])
    run = asyncio.create_task(g.compile().ainvoke({"n": 0}))
    await all_started.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert cancelled == 5


async def test_max_concurrency_caps_parallel_tasks() -> None:
    running = 0
    peak = 0

    async def work(item: int) -> dict[str, Any]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return {"log": [str(item)]}

    g = StateGraph(State)
    g.add_node("work", work)
    g.add_conditional_edges(START, lambda s: [Send("work", i) for i in range(12)], ["work"])
    result = await g.compile().ainvoke({"n": 0}, {"max_concurrency": 3})
    assert peak == 3
    assert len(result["log"]) == 12
