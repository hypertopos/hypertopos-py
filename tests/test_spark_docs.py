# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""docs/spark.md examples are code: they compile, import what exists, and the try-it block runs."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

SPARK_MD = Path(__file__).parents[1] / "docs" / "spark.md"
BLOCKS = re.findall(r"```python\n(.*?)```", SPARK_MD.read_text(encoding="utf-8"), flags=re.S)


def test_every_python_block_compiles_and_imports_real_names():
    assert BLOCKS, "spark.md has no python examples"
    for block in BLOCKS:
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hypertopos"):
                module = importlib.import_module(node.module or "")
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name} in spark.md"


def test_try_it_block_runs(spark, tmp_path, monkeypatch):
    pytest.importorskip("pyspark")
    [block] = [b for b in BLOCKS if "synthetic_aml(" in b and "find_anomalies(" in b]
    # The same code on a smaller population, to keep the test quick.
    code = block.replace("n_accounts=5_000", "n_accounts=600")
    assert code != block
    monkeypatch.setenv("HYPERTOPOS_SPHERE_ROOT", str(tmp_path))
    namespace: dict[str, object] = {"spark": spark}
    exec(compile(code, "spark.md", "exec"), namespace)  # noqa: S102 - the doc example itself
    result = namespace["result"]
    assert result.count() > 0  # type: ignore[attr-defined]


GDS_SPARK_SKILL = Path(__file__).parents[2] / "hypertopos-skills" / "gds-spark" / "SKILL.md"


@pytest.mark.skipif(not GDS_SPARK_SKILL.exists(), reason="skills package not in this checkout")
def test_gds_spark_skill_calls_use_real_names_and_keywords():
    import inspect

    blocks = re.findall(r"```python\n(.*?)```", GDS_SPARK_SKILL.read_text(encoding="utf-8"), re.S)
    assert blocks, "gds-spark has no python examples"
    for block in blocks:
        tree = ast.parse(block)
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hypertopos"):
                module = importlib.import_module(node.module or "")
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name} in gds-spark"
                    functions[alias.name] = getattr(module, alias.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fn = functions.get(node.func.id)
                if fn is None or not callable(fn):
                    continue
                params = inspect.signature(fn).parameters
                for kw in node.keywords:
                    assert kw.arg in params, f"{node.func.id}({kw.arg}=) in gds-spark"
