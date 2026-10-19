"""Errors raised by the graph runner."""


class GraphError(Exception):
    """Base class for every error raised by bhoot_graph itself."""


class InvalidGraphError(GraphError):
    """The graph is wired wrongly; raised by ``add_*`` and ``compile()``."""


class InvalidUpdateError(GraphError):
    """A node, router or input produced an update the state cannot accept."""


class GraphRecursionError(GraphError):
    """The run hit ``recursion_limit`` supersteps without reaching END."""


class GraphTimeoutError(GraphError):
    """A superstep took longer than ``step_timeout``."""
