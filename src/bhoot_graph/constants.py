"""Reserved node names."""

from typing import Final

START: Final = "__start__"
END: Final = "__end__"

RESERVED: Final = frozenset({START, END})
