"""State channels: how updates from nodes are merged into the state.

A key annotated ``Annotated[T, reducer]`` folds every write through ``reducer(old, new)``, so
parallel branches can all contribute. Any other key keeps the last value, and two writes to
it in the same superstep are an error: the order between parallel branches is not something
a graph should depend on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, Required, get_args, get_origin, get_type_hints

from bhoot_graph.errors import InvalidGraphError, InvalidUpdateError

Reducer = Callable[[Any, Any], Any]


@dataclass(frozen=True, slots=True)
class Channel:
    key: str
    reducer: Reducer | None = None
    empty: Callable[[], Any] | None = None  # reducer start value, e.g. list for list + add


def channels_for(schema: type) -> dict[str, Channel]:
    """Read the channels from a state schema (a TypedDict or any annotated class)."""
    try:
        hints = get_type_hints(schema, include_extras=True)
    except Exception as exc:
        raise InvalidGraphError(
            f"cannot read the annotations of {schema!r}: {exc} (reducers and types used in "
            "the schema must be importable from its module, not defined inside a function)"
        ) from exc
    if not hints:
        raise InvalidGraphError(f"state schema {schema!r} declares no keys")
    return {key: _channel(key, hint) for key, hint in hints.items()}


def _channel(key: str, hint: Any) -> Channel:
    while get_origin(hint) in (Required, NotRequired):
        hint = get_args(hint)[0]
    if get_origin(hint) is not Annotated:
        return Channel(key)
    base, *metadata = get_args(hint)
    reducers = [m for m in metadata if callable(m) and not isinstance(m, type)]
    if not reducers:
        return Channel(key)
    return Channel(key, reducers[-1], _empty_factory(base))


def _empty_factory(base: Any) -> Callable[[], Any] | None:
    factory = get_origin(base) or base
    if not isinstance(factory, type):
        return None
    try:
        factory()
    except Exception:  # types without a no-argument constructor start from the first write
        return None
    return factory


def apply_writes(
    state: Mapping[str, Any],
    channels: Mapping[str, Channel],
    writes: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return a new state with ``writes`` applied in the given order.

    ``writes`` are ``(writer, update)`` pairs already sorted by task, which makes the merge
    deterministic no matter which branch finished first.
    """
    new = dict(state)
    written_by: dict[str, str] = {}
    for writer, update in writes:
        for key, value in update.items():
            channel = channels.get(key)
            if channel is None:
                raise InvalidUpdateError(
                    f"{writer} wrote unknown key {key!r}; state keys are {sorted(channels)}"
                )
            if channel.reducer is None:
                if key in written_by:
                    raise InvalidUpdateError(
                        f"{key!r} was written by both {written_by[key]} and {writer} in one "
                        "step; give it a reducer, e.g. Annotated[list, operator.add]"
                    )
                written_by[key] = writer
                new[key] = value
                continue
            if key in new:
                new[key] = channel.reducer(new[key], value)
            elif channel.empty is not None:
                new[key] = channel.reducer(channel.empty(), value)
            else:
                new[key] = value
    return new
