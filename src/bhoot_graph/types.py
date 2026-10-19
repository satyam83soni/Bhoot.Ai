"""Values nodes and routers return to steer the graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Send:
    """Run ``node`` in the next superstep with ``arg`` as its input instead of the state.

    Returned by a router (or in ``Command.goto``) once per item to fan out; each Send is its
    own task, so ten Sends to the same node run it ten times in parallel.
    """

    node: str
    arg: Any


@dataclass(frozen=True, slots=True)
class Command:
    """Returned by a node to update the state and choose where to go next in one step.

    ``goto`` adds to the node's static edges; it does not replace them.
    """

    update: Mapping[str, Any] | None = None
    goto: str | Send | Sequence[str | Send] = field(default=())
