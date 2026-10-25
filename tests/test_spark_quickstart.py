# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""docs/examples/spark-quickstart.ipynb is code: its cells compile and run end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "docs" / "examples" / "spark-quickstart.ipynb"


def _code() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    lines = [
        line
        for cell in cells
        if cell["cell_type"] == "code"
        for line in "".join(cell["source"]).splitlines()
        if not line.lstrip().startswith("%")  # notebook magics such as %pip
    ]
    return "\n".join(lines)


def test_notebook_compiles():
    compile(_code(), "spark-quickstart.ipynb", "exec")


def test_notebook_runs(spark, tmp_path, monkeypatch):
    pytest.importorskip("pyspark")
    # The same cells on a smaller population, to keep the test quick.
    code = _code().replace("n_accounts=5_000", "n_accounts=600")
    assert code.count("n_accounts=600") == 2
    monkeypatch.setenv("HYPERTOPOS_SPHERE_ROOT", str(tmp_path))
    namespace: dict[str, object] = {"spark": spark}
    exec(compile(code, "spark-quickstart.ipynb", "exec"), namespace)  # noqa: S102 - the notebook itself
    assert namespace["scored"].count() > 0  # type: ignore[attr-defined]
    assert namespace["geometry"].num_rows > 0  # type: ignore[attr-defined]
