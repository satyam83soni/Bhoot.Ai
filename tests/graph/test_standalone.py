"""bhoot_graph must stay a standalone package that needs only the standard library."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import bhoot_graph

PACKAGE = Path(bhoot_graph.__file__).parent


def test_imports_only_the_standard_library_and_itself() -> None:
    outside: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top != "bhoot_graph" and top not in sys.stdlib_module_names:
                    outside.add(f"{path.name}: {name}")
    assert not outside, sorted(outside)
