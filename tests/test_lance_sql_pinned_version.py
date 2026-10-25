# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""SQL pushdown paths read the geometry version the session pinned.

A session pins each geometry dataset's Lance version when it opens, so a
sphere that is written to during the session (an ingest, a recalibration)
keeps serving one consistent snapshot. The SQL pushdown paths opened the
geometry by path and read the latest version instead.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest

from hypertopos.engine import lance_sql_agg as lsa

PACKAGE = Path(__file__).resolve().parents[1] / "hypertopos"
PATTERN = SimpleNamespace(
    relations=[SimpleNamespace(line_id="acc"), SimpleNamespace(line_id="cur")]
)


def _geometry(rows: list[tuple[str, str, str, float]]) -> pa.Table:
    return pa.table(
        {
            "primary_key": [r[0] for r in rows],
            "entity_keys": [[r[1], r[2]] for r in rows],
            "delta_norm": pa.array([r[3] for r in rows], pa.float32()),
        }
    )


@pytest.fixture(scope="module")
def paths(tmp_path_factory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("pinned")
    geo = str(root / "geo.lance")
    v1_rows = [("E1", "A", "USD", 1.0), ("E2", "A", "EUR", 2.0), ("E3", "B", "USD", 3.0)]
    v2_rows = v1_rows + [("E4", "C", "EUR", 9.0), ("E5", "C", "USD", 8.0)]
    lance.write_dataset(_geometry(v1_rows), geo)
    v1 = lance.dataset(geo).version
    lance.write_dataset(_geometry(v2_rows), geo, mode="overwrite")
    events = str(root / "events.lance")
    lance.write_dataset(
        pa.table(
            {
                "primary_key": ["E1", "E2", "E3", "E4", "E5"],
                "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
                "channel": ["web", "web", "app", "app", "web"],
            }
        ),
        events,
    )
    props = str(root / "accounts.lance")
    lance.write_dataset(
        pa.table({"primary_key": ["A", "B", "C"], "region": ["n", "s", "s"]}), props
    )
    return {"geo": geo, "v1": v1, "events": events, "props": props}


def _calls(p: dict[str, object]) -> dict[str, object]:
    geo, events, props = p["geo"], p["events"], p["props"]
    return {
        "count": lambda **kw: lsa.aggregate_count(geo, PATTERN, "acc", **kw),
        "metric": lambda **kw: lsa.aggregate_metric(
            geo, events, PATTERN, "acc", "amount", "sum", **kw
        ),
        "filtered_metric": lambda **kw: lsa.aggregate_filtered_metric(
            geo, events, PATTERN, "acc", "amount", "sum", [("cur", {"USD", "EUR"})], "events", **kw
        ),
        "percentile": lambda **kw: lsa.aggregate_percentile(
            geo, events, PATTERN, "acc", "amount", 0.5, **kw
        ),
        "pivot": lambda **kw: lsa.aggregate_pivot(geo, events, PATTERN, "acc", "channel", **kw),
        "property": lambda **kw: lsa.aggregate_property(
            geo, props, PATTERN, "cur", "acc", "region", **kw
        ),
        "find_anomalies": lambda **kw: lsa.find_anomalies(geo, threshold=0.0, top_n=10, **kw),
    }


V1_EXPECTED = {
    "count": {"A": 2, "B": 1},
    "find_anomalies_keys": ["E3", "E2", "E1"],
}


@pytest.mark.parametrize(
    "name",
    ["count", "metric", "filtered_metric", "percentile", "pivot", "property", "find_anomalies"],
)
def test_pinned_version_differs_from_latest(paths: dict[str, object], name: str) -> None:
    call = _calls(paths)[name]
    pinned = call(lance_version=paths["v1"])
    latest = call()
    assert pinned != latest, f"{name}: pinned and latest reads must differ on this fixture"
    assert call(lance_version=None) == latest


def test_pinned_results_match_the_pinned_snapshot(paths: dict[str, object]) -> None:
    calls = _calls(paths)
    assert calls["count"](lance_version=paths["v1"]) == V1_EXPECTED["count"]
    assert (
        calls["find_anomalies"](lance_version=paths["v1"])["keys"]
        == V1_EXPECTED["find_anomalies_keys"]
    )
    metrics, counts = calls["metric"](lance_version=paths["v1"])
    assert metrics == {"A": 30.0, "B": 30.0}
    assert counts == {"A": 2, "B": 1}


def _sql_calls(rel: str) -> list[ast.Call]:
    tree = ast.parse((PACKAGE / rel).read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_lance_sql_")
    ]


@pytest.mark.parametrize("rel", ["engine/aggregation.py", "navigation/navigator.py"])
def test_every_sql_pushdown_call_passes_the_pinned_version(rel: str) -> None:
    calls = _sql_calls(rel)
    assert calls, f"no SQL pushdown calls found in {rel}"
    missing = [c.lineno for c in calls if not any(k.arg == "lance_version" for k in c.keywords)]
    assert not missing, f"{rel}: SQL pushdown calls without lance_version at lines {missing}"
