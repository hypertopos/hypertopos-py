# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Ratchet: read paths open Lance datasets only through the session's reader.

The reader applies the session pin (and later the session's branch); a direct
``lance.dataset`` / ``open_dataset`` call in the navigator or the engine reads
whatever is latest on disk.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "hypertopos"
READ_PATH_FILES = sorted(
    {*PACKAGE.glob("navigation/*.py"), *PACKAGE.glob("engine/*.py")}
    - {PACKAGE / "engine" / "lance_sql_agg.py"}
) + [
    PACKAGE / "storage" / "topology_cache.py",
    PACKAGE / "storage" / "calibration_history.py",
    PACKAGE / "sphere.py",
]
SQL_AGG_ALLOWED = {"_open_geometry", "_open_points"}


def _direct_opens(tree: ast.AST) -> list[tuple[str, int]]:
    """(enclosing function, line) of every open_dataset / *.dataset call."""
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, fn: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else fn
            if isinstance(child, ast.Call):
                func = child.func
                if (isinstance(func, ast.Name) and func.id == "open_dataset") or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "dataset"
                    and isinstance(func.value, ast.Name)
                    and func.value.id.lstrip("_").startswith("lance")
                ):
                    found.append((fn, child.lineno))
            visit(child, name)

    visit(tree, "<module>")
    return found


@pytest.mark.parametrize("path", READ_PATH_FILES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_read_path_module_opens_nothing_directly(path: Path) -> None:
    opens = _direct_opens(ast.parse(path.read_text(encoding="utf-8")))
    assert not opens, f"direct Lance opens (use GDSReader.dataset): {opens}"


def test_sql_pushdown_opens_only_through_its_helpers() -> None:
    path = PACKAGE / "engine" / "lance_sql_agg.py"
    opens = _direct_opens(ast.parse(path.read_text(encoding="utf-8")))
    stray = [(fn, line) for fn, line in opens if fn not in SQL_AGG_ALLOWED]
    assert not stray, f"open_dataset outside {sorted(SQL_AGG_ALLOWED)}: {stray}"
