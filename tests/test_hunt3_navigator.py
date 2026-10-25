# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Navigator regressions: motif ranking cache, chains, paths, drift ranking."""

from __future__ import annotations

from typing import Any

import pytest

from hypertopos.sphere import HyperSphere
from tests.conftest import FIXTURES_PATH


@pytest.fixture
def nav() -> Any:
    return HyperSphere.open(str(FIXTURES_PATH)).session("hunt3").navigator()


def test_motif_ranking_cache_is_keyed_by_min_k(nav: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def fake_rank(
        seeds: list[str], pattern_id: str, motif_type: str, window: Any, min_k: Any, **_: Any
    ) -> list[dict[str, Any]]:
        calls.append(min_k)
        return [{"seed": s, "min_k": min_k} for s in seeds]

    monkeypatch.setattr(nav, "_require_anchor_pattern_for_motif", lambda _pid: None)
    monkeypatch.setattr(nav, "_rank_motifs", fake_rank)
    first = nav.find_high_potential_motifs("customer_pattern", "fan_out", top_n=3)
    second = nav.find_high_potential_motifs("customer_pattern", "fan_out", top_n=3, min_k=5)
    assert calls == [None, 5]
    assert {r["min_k"] for r in first} == {None}
    assert {r["min_k"] for r in second} == {5}


def test_drift_ranking_honours_a_dimension_with_a_display_name(tmp_path: Any) -> None:
    import json

    from tests.conftest import clone_sphere

    sphere = clone_sphere(FIXTURES_PATH, tmp_path / "s")
    semantics = {
        "patterns": {
            "customer_pattern": {"relations": {"products": {"display_name": "bought products"}}}
        }
    }
    (sphere / "_gds_meta" / "semantics.json").write_text(json.dumps(semantics), encoding="utf-8")
    nav = HyperSphere.open(str(sphere)).session("hunt3").navigator()
    for name in ("products", "bought products"):
        ranked = nav.π9_attract_drift("customer_pattern", top_n=10, rank_by_dimension=name)
        diffs = [abs(r["dimension_diffs"]["bought products"]) for r in ranked]
        assert diffs == sorted(diffs, reverse=True), name


def test_collective_drift_survives_entities_drifting_identically(
    nav: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every entity drifts in exactly the same direction: k-means++ seeding
    # sees zero distance to the first centre for all points.
    same = [
        {"primary_key": f"E{i}", "dimension_diffs": {"a": 1.0, "b": 0.0}, "displacement": 1.0}
        for i in range(12)
    ]
    monkeypatch.setattr(nav, "π9_attract_drift", lambda *a, **k: same)
    clusters = nav.detect_collective_drift("customer_pattern", n_clusters=3, min_cluster_size=2)
    assert isinstance(clusters, list)


def test_regime_change_from_cache_explains_empty_results(nav: Any) -> None:
    from datetime import UTC, datetime, timedelta

    start = datetime(2024, 1, 1, tzinfo=UTC)

    def window(i: int, centroid: list[float]) -> dict[str, Any]:
        return {
            "window_start": start + timedelta(days=i),
            "window_end": start + timedelta(days=i + 1),
            "centroid": centroid,
        }

    one = nav._pi12_from_cache("customer_pattern", [window(0, [0.0, 0.0])], None, None, 5)
    assert "warning" in one[0]
    flat = [window(i, [1.0, 1.0]) for i in range(5)]
    none_found = nav._pi12_from_cache("customer_pattern", flat, None, None, 5)
    assert none_found[0]["warning"].startswith("no_regime_changes_detected")


def test_trajectory_scans_sample_keys_and_keep_whole_trajectories(
    nav: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = nav._storage.read_temporal_batched

    def spy(pattern_id: str, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("keys"))
        return real(pattern_id, *args, **kwargs)

    monkeypatch.setattr(nav._storage, "read_temporal_batched", spy)
    nav.classify_trajectory("CUST-001", "customer_pattern", sample_size=1)
    nav.detect_trajectory_anomaly("customer_pattern", sample_size=1)
    assert seen[0] is not None and "CUST-001" in seen[0], seen
    assert seen[1] is not None and len(seen[1]) == 1, seen
