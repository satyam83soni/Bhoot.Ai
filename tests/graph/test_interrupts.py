"""bhoot_graph human in the loop: interrupt(), Command(resume=...), deadlines, bad answers."""

from __future__ import annotations

import asyncio
import operator
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest

from bhoot_graph import (
    START,
    BaseCheckpointSaver,
    Command,
    GraphError,
    Interrupt,
    MemorySaver,
    Send,
    SqliteSaver,
    StateGraph,
    interrupt,
)


class State(TypedDict):
    total: str
    log: Annotated[list[str], operator.add]


CONFIG = {"configurable": {"thread_id": "job-7"}}


@pytest.fixture(params=["memory", "sqlite"])
def saver(request: pytest.FixtureRequest, tmp_path: Path) -> BaseCheckpointSaver:
    return MemorySaver() if request.param == "memory" else SqliteSaver(tmp_path / "hitl.db")


def _review_graph(saver: BaseCheckpointSaver, calls: Counter[str]) -> Any:
    async def extract(state: State) -> dict[str, Any]:
        calls["extract"] += 1
        return {"total": "1,25O.00", "log": ["extract"]}

    async def review(state: State) -> dict[str, Any]:
        calls["review"] += 1
        answer = interrupt({"field": "total", "value": state["total"], "question": "correct?"})
        return {"total": answer["total"], "log": ["review"]}

    g = StateGraph(State)
    g.add_node("extract", extract)
    g.add_node("review", review)
    g.add_edge(START, "extract")
    g.add_edge("extract", "review")
    return g.compile(saver)


# ---------------------------------------------------------------- pause and resume
async def test_pause_ask_and_resume_with_the_answer(saver: BaseCheckpointSaver) -> None:
    calls: Counter[str] = Counter()
    app = _review_graph(saver, calls)
    paused = await app.ainvoke({"log": []}, CONFIG)
    (question,) = paused["__interrupt__"]
    assert isinstance(question, Interrupt)
    assert question.value["value"] == "1,25O.00"
    snapshot = await app.aget_state(CONFIG)
    assert snapshot is not None
    assert snapshot.next == ("review",)
    assert snapshot.interrupts == (question,)

    done = await app.ainvoke(Command(resume={"total": "1,250.00"}), CONFIG)
    assert done == {"total": "1,250.00", "log": ["extract", "review"]}
    assert calls == {"extract": 1, "review": 2}  # only the paused node ran again
    after = await app.aget_state(CONFIG)
    assert after is not None
    assert after.interrupts == ()


async def test_except_exception_in_a_node_does_not_swallow_the_pause() -> None:
    async def careful(state: State) -> dict[str, Any]:
        try:
            answer = interrupt("ok?")
        except Exception:  # an everyday safety net in node code
            answer = "swallowed"
        return {"total": answer}

    g = StateGraph(State)
    g.add_node("careful", careful)
    g.add_edge(START, "careful")
    app = g.compile(MemorySaver())
    assert "__interrupt__" in await app.ainvoke({}, CONFIG)
    assert (await app.ainvoke(Command(resume="yes"), CONFIG))["total"] == "yes"


async def test_several_questions_in_one_node_are_answered_in_order(
    saver: BaseCheckpointSaver,
) -> None:
    def ask(state: State) -> dict[str, Any]:  # a sync node, run in a thread
        bank = interrupt("bank name?")
        branch = interrupt(f"branch of {bank}?")
        return {"total": f"{bank}/{branch}"}

    g = StateGraph(State)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    app = g.compile(saver)
    first = await app.ainvoke({}, CONFIG)
    assert first["__interrupt__"][0].value == "bank name?"
    second = await app.ainvoke(Command(resume="SBI"), CONFIG)
    assert second["__interrupt__"][0].value == "branch of SBI?"
    assert second["__interrupt__"][0].id.endswith("#1")
    final = await app.ainvoke(Command(resume="Pune"), CONFIG)
    assert final["total"] == "SBI/Pune"


async def test_pause_during_fan_out_keeps_the_finished_siblings(
    saver: BaseCheckpointSaver,
) -> None:
    class Fan(TypedDict):
        out: Annotated[list[str], operator.add]

    calls: Counter[int] = Counter()

    async def page(n: int) -> dict[str, Any]:
        calls[n] += 1
        if n == 2:
            return {"out": [f"p2={interrupt('page 2 is blurry, value?')}"]}
        return {"out": [f"p{n}"]}

    g = StateGraph(Fan)
    g.add_node("page", page)
    g.add_conditional_edges(START, lambda s: [Send("page", n) for n in (1, 2, 3)], ["page"])
    app = g.compile(saver)
    await app.ainvoke({}, CONFIG)
    result = await app.ainvoke(Command(resume="42"), CONFIG)
    assert result["out"] == ["p1", "p2=42", "p3"]
    assert calls == {1: 1, 2: 2, 3: 1}


async def test_several_pending_questions_are_answered_by_id(saver: BaseCheckpointSaver) -> None:
    class Fan(TypedDict):
        out: Annotated[list[str], operator.add]

    async def field(name: str) -> dict[str, Any]:
        return {"out": [f"{name}={interrupt(name)}"]}

    g = StateGraph(Fan)
    g.add_node("field", field)
    g.add_conditional_edges(START, lambda s: [Send("field", n) for n in ("a", "b")], ["field"])
    app = g.compile(saver)
    questions = (await app.ainvoke({}, CONFIG))["__interrupt__"]
    by_value = {q.value: q.id for q in questions}
    with pytest.raises(GraphError, match="2 interrupts are pending"):
        await app.ainvoke(Command(resume="x"), CONFIG)

    partly = await app.ainvoke(Command(resume={by_value["a"]: "1"}), CONFIG)
    assert [q.value for q in partly["__interrupt__"]] == ["b"]  # "a" is answered, "b" still waits
    done = await app.ainvoke(Command(resume={by_value["b"]: "2"}), CONFIG)
    assert done["out"] == ["a=1", "b=2"]


async def test_resuming_without_an_answer_asks_again(saver: BaseCheckpointSaver) -> None:
    calls: Counter[str] = Counter()
    app = _review_graph(saver, calls)
    first = (await app.ainvoke({"log": []}, CONFIG))["__interrupt__"]
    again = (await app.ainvoke(None, CONFIG))["__interrupt__"]
    assert again == first


async def test_an_answer_survives_a_crash_after_it_was_given(saver: BaseCheckpointSaver) -> None:
    crash = {"on": True}

    async def review(state: State) -> dict[str, Any]:
        answer = interrupt("total?")
        if crash["on"]:
            raise RuntimeError("worker died")
        return {"total": answer}

    g = StateGraph(State)
    g.add_node("review", review)
    g.add_edge(START, "review")
    app = g.compile(saver)
    await app.ainvoke({}, CONFIG)
    with pytest.raises(RuntimeError, match="worker died"):
        await app.ainvoke(Command(resume="900.00"), CONFIG)
    crash["on"] = False
    assert (await app.ainvoke(None, CONFIG))["total"] == "900.00"  # no need to ask again


# ---------------------------------------------------------------- deadlines
async def test_deadline_passes_and_the_default_is_used(saver: BaseCheckpointSaver) -> None:
    deadline = datetime.now(UTC) + timedelta(milliseconds=50)

    async def review(state: State) -> dict[str, Any]:
        return {"total": interrupt("total?", deadline=deadline, default="needs_review")}

    g = StateGraph(State)
    g.add_node("review", review)
    g.add_edge(START, "review")
    app = g.compile(saver)
    paused = await app.ainvoke({}, CONFIG)
    assert paused["__interrupt__"][0].deadline == deadline
    await asyncio.sleep(0.1)
    assert (await app.ainvoke(None, CONFIG))["total"] == "needs_review"


async def test_deadline_arguments_are_checked() -> None:
    async def naive(state: State) -> dict[str, Any]:
        return {"total": interrupt("x", deadline=datetime(2030, 1, 1), default="d")}

    async def no_default(state: State) -> dict[str, Any]:
        return {"total": interrupt("x", deadline=datetime.now(UTC))}

    for node, message in [(naive, "timezone-aware"), (no_default, "needs a default")]:
        g = StateGraph(State)
        g.add_node("n", node)
        g.add_edge(START, "n")
        with pytest.raises(GraphError, match=message):
            await g.compile(MemorySaver()).ainvoke({}, CONFIG)


# ---------------------------------------------------------------- misuse
async def test_bad_resumes_are_rejected(saver: BaseCheckpointSaver) -> None:
    calls: Counter[str] = Counter()
    app = _review_graph(saver, calls)
    with pytest.raises(GraphError, match="paused thread"):
        await app.ainvoke(Command(resume="x"), {"configurable": {"thread_id": "never-ran"}})
    question = (await app.ainvoke({"log": []}, CONFIG))["__interrupt__"][0]
    with pytest.raises(GraphError, match="no pending interrupt with id"):
        await app.ainvoke(Command(resume={question.id: {"total": "1"}, "stale#9": 1}), CONFIG)
    with pytest.raises(GraphError, match="only carry resume"):
        await app.ainvoke(Command(update={"total": "1"}), CONFIG)
    await app.ainvoke(Command(resume={"total": "1"}), CONFIG)
    with pytest.raises(GraphError, match="no pending interrupt to answer"):
        await app.ainvoke(Command(resume={"total": "2"}), CONFIG)  # answering twice


async def test_interrupt_needs_a_checkpointer_and_a_node() -> None:
    g = StateGraph(State)
    g.add_node("ask", lambda s: {"total": interrupt("?")})
    g.add_edge(START, "ask")
    with pytest.raises(GraphError, match="needs a checkpointer"):
        await g.compile().ainvoke({})
    with pytest.raises(GraphError, match="inside a graph node"):
        interrupt("not in a node")


async def test_pause_is_streamed(saver: BaseCheckpointSaver) -> None:
    calls: Counter[str] = Counter()
    app = _review_graph(saver, calls)
    events = [e async for e in app.astream({"log": []}, CONFIG, stream_mode=["updates", "tasks"])]
    updates = [chunk for mode, chunk in events if mode == "updates"]
    assert "__interrupt__" in updates[-1]
    results = [c for mode, c in events if mode == "tasks" and c["type"] == "task_result"]
    assert results[-1]["node"] == "review"
    assert results[-1]["error"] == "interrupted"
