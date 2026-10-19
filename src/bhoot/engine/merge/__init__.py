"""Merge: combine answers from every call into one value per field."""

from bhoot.engine.merge.merger import Merger, MergeResult, array_root, initial_states
from bhoot.engine.merge.strategies import (
    LIST_STRATEGIES,
    SCALAR_STRATEGIES,
    ItemOrder,
    ListConcat,
    RowsByPage,
    ScalarPreferVerified,
)

__all__ = [
    "LIST_STRATEGIES",
    "SCALAR_STRATEGIES",
    "ItemOrder",
    "ListConcat",
    "MergeResult",
    "Merger",
    "RowsByPage",
    "ScalarPreferVerified",
    "array_root",
    "initial_states",
]
