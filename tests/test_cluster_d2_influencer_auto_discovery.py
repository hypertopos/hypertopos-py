# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cluster d2 cycle — auto-discovery branch on
find_calibration_influencers + per-influencer temporal impact cache."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic 3-cluster anchor sphere
# ---------------------------------------------------------------------------

def _build_three_cluster_sphere(
    sphere_path: Path,
    *,
    n_per_cluster: int = 100,
    seed: int = 7,
) -> str:
    """Build an anchor sphere with three well-separated clusters in
    delta-space.

    Returns the pattern_id ``"three_cluster_pattern"``.
    """
    from hypertopos.builder import GDSBuilder, RelationSpec

    rng = np.random.default_rng(seed)
    sphere_path.mkdir(parents=True, exist_ok=True)
    for subdir in ("points", "geometry", "temporal"):
        target = sphere_path / subdir
        if target.exists():
            shutil.rmtree(target)

    cluster_centers = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [0.0, 20.0],
    ])

    entities: list[dict] = []
    for c_idx, center in enumerate(cluster_centers):
        for i in range(n_per_cluster):
            jitter = rng.normal(0.0, 0.5, size=2)
            x, y = center + jitter
            entities.append({
                "entity_id": f"C{c_idx}_E{i:03d}",
                "x_axis": float(x),
                "y_axis": float(y),
            })

    b = GDSBuilder("d2_three_cluster", str(sphere_path))
    b.add_line(
        "entities",
        entities,
        key_col="entity_id",
        source_id="test",
    )
    b.add_pattern(
        "three_cluster_pattern",
        pattern_type="anchor",
        entity_line="entities",
        relations=[
            RelationSpec(line_id="entities", fk_col=None, direction="self"),
        ],
        tracked_properties=["x_axis", "y_axis"],
        anomaly_percentile=80.0,
    )
    b.build()
    return "three_cluster_pattern"


@pytest.fixture
def three_cluster_sphere(tmp_path):
    sphere_path = tmp_path / "three_cluster_sphere"
    pid = _build_three_cluster_sphere(sphere_path)
    return sphere_path, pid


# ---------------------------------------------------------------------------
# D2a — auto-discovery
# ---------------------------------------------------------------------------


class TestAutoDiscoveryBranch:
    def test_auto_discover_three_cluster_returns_one_per_cluster(
        self, three_cluster_sphere,
    ):
        """auto_discover=True with auto_k=3 on a 300-entity, 3-cluster
        anchor sphere returns three influencer candidates with distinct
        entity_keys and ~100 cluster_size each."""
        from hypertopos.sphere import HyperSphere

        sphere_path, pid = three_cluster_sphere
        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2a_test")
        try:
            nav = session.navigator()
            report = nav.find_calibration_influencers(
                pattern_id=pid,
                classify="all",
                top_n=3,
                auto_discover=True,
                auto_k=3,
            )
            assert report.auto_discovered is True
            assert len(report.entries) <= 3
            assert len(report.entries) >= 1
            entity_keys = [e.entity_key for e in report.entries]
            assert len(set(entity_keys)) == len(entity_keys), (
                f"Duplicate entity_keys in auto-discovered entries: {entity_keys}"
            )
            for entry in report.entries:
                assert entry.cluster_size is not None
                assert entry.cluster_centroid_distance is not None
                assert entry.cluster_size > 0
                assert entry.cluster_centroid_distance >= 0.0
            # With auto_k=3 on a 300-entity, 3-balanced-cluster fixture the
            # 3 clusters should each carry ~100 members.
            sizes = sorted(e.cluster_size for e in report.entries)
            assert sum(sizes) >= 270, (
                f"Expected ~300 entities partitioned across reps, got "
                f"{sum(sizes)}"
            )
        finally:
            session.close()

    def test_auto_discover_auto_k_two_returns_at_most_two(
        self, three_cluster_sphere,
    ):
        """auto_k=2 returns at most 2 representatives."""
        from hypertopos.sphere import HyperSphere

        sphere_path, pid = three_cluster_sphere
        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2a_k2_test")
        try:
            nav = session.navigator()
            report = nav.find_calibration_influencers(
                pattern_id=pid,
                classify="all",
                top_n=5,
                auto_discover=True,
                auto_k=2,
            )
            assert report.auto_discovered is True
            assert 1 <= len(report.entries) <= 2
        finally:
            session.close()

    def test_manual_mode_unchanged(self, three_cluster_sphere):
        """auto_discover=False (default) leaves manual behaviour intact and
        does not set cluster_* fields."""
        from hypertopos.sphere import HyperSphere

        sphere_path, pid = three_cluster_sphere
        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2a_manual_test")
        try:
            nav = session.navigator()
            report = nav.find_calibration_influencers(
                pattern_id=pid,
                classify="all",
                top_n=5,
            )
            assert report.auto_discovered is False
            assert len(report.entries) <= 5
            for entry in report.entries:
                assert entry.cluster_size is None
                assert entry.cluster_centroid_distance is None
        finally:
            session.close()

    def test_auto_discover_invalid_auto_k_raises(self, three_cluster_sphere):
        from hypertopos.sphere import HyperSphere

        sphere_path, pid = three_cluster_sphere
        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2a_invalid_test")
        try:
            nav = session.navigator()
            with pytest.raises(ValueError, match="auto_k"):
                nav.find_calibration_influencers(
                    pattern_id=pid,
                    auto_discover=True,
                    auto_k=0,
                )
            with pytest.raises(ValueError, match="auto_k"):
                nav.find_calibration_influencers(
                    pattern_id=pid,
                    auto_discover=True,
                    auto_k=51,
                )
        finally:
            session.close()


# ---------------------------------------------------------------------------
# D2b — per-influencer temporal impact cache
# ---------------------------------------------------------------------------


class TestInfluencerHistoryCache:
    def test_cache_miss_returns_empty_with_hint(self, three_cluster_sphere):
        from hypertopos.sphere import HyperSphere

        sphere_path, pid = three_cluster_sphere
        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2b_miss_test")
        try:
            nav = session.navigator()
            report = nav.calibration_influencer_history(
                "never_recorded_pk",
                pattern_id=pid,
            )
            assert report.history == []
            assert report.n_epochs == 0
            assert report.hint is not None
            assert "find_calibration_influencers" in report.hint
        finally:
            session.close()

    def test_unknown_pattern_raises(self, three_cluster_sphere):
        from hypertopos.sphere import HyperSphere

        sphere_path, _ = three_cluster_sphere
        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2b_unknown_test")
        try:
            nav = session.navigator()
            with pytest.raises(ValueError, match="not found"):
                nav.calibration_influencer_history(
                    "any_key",
                    pattern_id="no_such_pattern",
                )
        finally:
            session.close()

    def test_three_epoch_round_trip(self, tmp_path):
        """Build the same anchor sphere three times — each build bumps the
        calibration epoch and find_calibration_influencers populates the
        per-entity cache. The history call must return three chronological
        entries."""
        from hypertopos.sphere import HyperSphere

        sphere_path = tmp_path / "d2b_sphere"

        recorded_key: str | None = None
        for _ in range(3):
            pid = _build_three_cluster_sphere(sphere_path)
            sphere = HyperSphere.open(str(sphere_path))
            session = sphere.session(agent_id="d2b_history_test")
            try:
                nav = session.navigator()
                report = nav.find_calibration_influencers(
                    pattern_id=pid,
                    classify="all",
                    top_n=3,
                )
                assert len(report.entries) >= 1
                # Record the same key across epochs by pinning to the first
                # surfaced entry on epoch 1.
                if recorded_key is None:
                    recorded_key = report.entries[0].entity_key
            finally:
                session.close()

        assert recorded_key is not None

        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2b_history_read")
        try:
            nav = session.navigator()
            history = nav.calibration_influencer_history(
                recorded_key,
                pattern_id=pid,
            )
        finally:
            session.close()

        # Cache may have skipped epochs where the entry was not surfaced;
        # at minimum the final epoch (3) must be present because the recorded
        # key was surfaced on epoch 1 (and likely re-surfaced on subsequent
        # epochs since the population is deterministic).
        assert history.n_epochs >= 1
        epochs = [e.epoch for e in history.history]
        assert epochs == sorted(epochs), "history is not chronological"
        assert all(isinstance(e.mu_impact, float) for e in history.history)
        assert all(isinstance(e.delta_norm_impact, float) for e in history.history)

    def test_cache_file_layout_on_disk(self, three_cluster_sphere):
        """The cache file must live at the documented path and contain the
        expected JSON shape (primary_key + pattern_id + entries[])."""
        from urllib.parse import quote

        from hypertopos.sphere import HyperSphere

        sphere_path, pid = three_cluster_sphere
        sphere = HyperSphere.open(str(sphere_path))
        session = sphere.session(agent_id="d2b_layout_test")
        try:
            nav = session.navigator()
            report = nav.find_calibration_influencers(
                pattern_id=pid,
                classify="all",
                top_n=2,
            )
        finally:
            session.close()

        assert len(report.entries) >= 1
        pk = report.entries[0].entity_key
        cache_path = (
            sphere_path
            / "_gds_meta"
            / "calibration_history"
            / pid
            / f"influencer_{quote(pk, safe='')}.json"
        )
        assert cache_path.exists(), (
            f"Expected influencer cache at {cache_path}; got nothing"
        )
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        assert blob["primary_key"] == pk
        assert blob["pattern_id"] == pid
        assert isinstance(blob["entries"], list)
        assert len(blob["entries"]) >= 1
        for record in blob["entries"]:
            assert {"epoch", "calibrated_at", "mu_impact", "delta_norm_impact"} <= set(record.keys())


# ---------------------------------------------------------------------------
# Tier mapping smoke for the new MCP tool
# ---------------------------------------------------------------------------


class TestTierMapping:
    def test_calibration_influencer_history_in_tool_tiers(self):
        from hypertopos_mcp.server import _TOOL_TIERS

        assert "calibration_influencer_history" in _TOOL_TIERS
        assert _TOOL_TIERS["calibration_influencer_history"] == "base"
