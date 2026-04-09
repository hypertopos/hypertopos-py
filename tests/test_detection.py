# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for detection methods — _classify_trajectory + navigator detectors."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.navigation.navigator import (
    GDSNavigator,
    GDSNavigationError,
    _classify_trajectory,
)


# -----------------------------------------------------------------------
# Pure function tests — _classify_trajectory
# -----------------------------------------------------------------------


class TestClassifyTrajectory:
    """Unit tests for the _classify_trajectory pure function."""

    def test_arch(self):
        """Rising then falling delta norms → arch."""
        # 10 points: low → peak in middle → low
        norms = [0.1, 0.3, 0.5, 0.8, 1.0, 0.9, 0.6, 0.4, 0.2, 0.1]
        assert _classify_trajectory(norms) == "arch"

    def test_v_shape(self):
        """Falling then rising delta norms → v_shape."""
        norms = [1.0, 0.8, 0.5, 0.2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        assert _classify_trajectory(norms) == "v_shape"

    def test_linear_drift(self):
        """Monotonically increasing norms → linear_drift."""
        norms = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert _classify_trajectory(norms) == "linear_drift"

    def test_flat(self):
        """Near-zero variation → flat."""
        norms = [0.5, 0.5, 0.5, 0.5, 0.5]
        assert _classify_trajectory(norms) == "flat"

    def test_spike_recovery(self):
        """Brief spike then return to baseline → spike_recovery."""
        norms = [0.1, 0.9, 0.7, 0.3, 0.2, 0.1, 0.05]
        assert _classify_trajectory(norms) == "spike_recovery"

    def test_monotonic_decrease_not_spike(self):
        """Monotonically decreasing (starts high) must NOT be spike_recovery."""
        norms = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        assert _classify_trajectory(norms) != "spike_recovery"

    def test_insufficient_data(self):
        """Fewer than 3 points → insufficient_data."""
        assert _classify_trajectory([0.1, 0.2]) == "insufficient_data"
        assert _classify_trajectory([]) == "insufficient_data"
        assert _classify_trajectory([0.5]) == "insufficient_data"


# -----------------------------------------------------------------------
# Navigator detection method tests (mocked)
# -----------------------------------------------------------------------


def _make_navigator() -> GDSNavigator:
    """Create a GDSNavigator with all dependencies mocked."""
    engine = MagicMock()
    storage = MagicMock()
    manifest = MagicMock()
    contract = MagicMock()
    nav = GDSNavigator(engine, storage, manifest, contract)
    return nav


@dataclass
class _FakeHit:
    primary_key: str
    score: int
    source_count: int = 1


@dataclass
class _FakeScanResult:
    hits: list[_FakeHit]


class TestDetectCrossPatternDiscrepancy:
    """Tests for detect_cross_pattern_discrepancy."""

    def test_single_pattern_returns_empty(self):
        """When scanner discovers < 2 sources, returns empty list."""
        nav = _make_navigator()

        fake_scanner = MagicMock()
        fake_scanner._sources = [MagicMock()]  # only 1 source
        fake_scanner.auto_discover.return_value = fake_scanner

        with patch(
            "hypertopos.navigation.scanner.PassiveScanner",
            return_value=fake_scanner,
        ):
            result = nav.detect_cross_pattern_discrepancy("accounts")

        assert result == []

    def test_filters_single_source_hits(self):
        """Only entities with score == 1 are included; those with higher scores are excluded."""
        nav = _make_navigator()

        hit_single = _FakeHit(primary_key="A001", score=1)
        hit_multi = _FakeHit(primary_key="A002", score=3)

        fake_scanner = MagicMock()
        fake_scanner._sources = [MagicMock(), MagicMock()]  # 2 sources
        fake_scanner.auto_discover.return_value = fake_scanner
        fake_scanner.scan.return_value = _FakeScanResult(hits=[hit_single, hit_multi])

        # cross_pattern_profile for A001: anomalous in pat_a, normal in pat_b
        nav.cross_pattern_profile = MagicMock(return_value={
            "primary_key": "A001",
            "line_id": "accounts",
            "source_count": 1,
            "total_patterns": 2,
            "signals": {
                "pat_a": {
                    "is_anomaly": True,
                    "delta_norm": 3.5,
                    "delta_rank_pct": 95.0,
                },
                "pat_b": {
                    "is_anomaly": False,
                    "delta_norm": 0.8,
                    "delta_rank_pct": 40.0,
                },
            },
        })

        with patch(
            "hypertopos.navigation.scanner.PassiveScanner",
            return_value=fake_scanner,
        ):
            result = nav.detect_cross_pattern_discrepancy("accounts")

        assert len(result) == 1
        assert result[0]["entity_key"] == "A001"
        assert result[0]["anomalous_pattern"] == "pat_a"
        assert result[0]["normal_patterns"] == ["pat_b"]
        assert result[0]["delta_norm_anomalous"] == 3.5


class TestDetectNeighborContamination:
    """Tests for detect_neighbor_contamination."""

    def test_empty_on_no_anomalies(self):
        """When geometry has no anomalous entities, returns empty."""
        nav = _make_navigator()
        nav._manifest.pattern_version.return_value = 1

        geo = pa.table({
            "primary_key": ["A001", "A002", "A003"],
            "is_anomaly": [False, False, False],
            "delta_rank_pct": [30.0, 50.0, 60.0],
        })
        nav._storage.read_geometry.return_value = geo

        result = nav.detect_neighbor_contamination("pat_a")
        assert result == []

    def test_returns_normal_targets_only(self):
        """Results always have is_anomaly_target=False (only normal seeds reported)."""
        nav = _make_navigator()
        nav._manifest.pattern_version.return_value = 1

        geo = pa.table({
            "primary_key": ["A001", "A002", "A003", "A004"],
            "is_anomaly": [True, True, False, False],
            "delta_rank_pct": [99.0, 98.0, 40.0, 50.0],
        })
        # First call: full geometry read, second call: neighbour batch read
        nav._storage.read_geometry.side_effect = [
            geo,
            # Batch read for neighbour anomaly status
            pa.table({
                "primary_key": ["A001", "A002", "A003"],
                "is_anomaly": [True, True, False],
            }),
        ]

        # find_similar_entities returns neighbours
        def fake_similar(pk, pat_id, top_n=10):
            return [("A001", 0.1), ("A002", 0.2), ("A003", 0.3)]

        nav.find_similar_entities = MagicMock(side_effect=fake_similar)

        with patch("random.sample", side_effect=lambda lst, n: lst[:n]):
            result = nav.detect_neighbor_contamination(
                "pat_a", k=3, sample_size=5,
                contamination_threshold=0.5,
            )

        # Only normal seeds are in results
        assert len(result) > 0
        for r in result:
            assert r["is_anomaly_target"] is False


    def test_inverted_search_finds_contaminated_normals(self):
        """Inverted search: start from anomalies, find their normal neighbors."""
        nav = _make_navigator()
        nav._manifest.pattern_version.return_value = 1

        # A001, A002 are anomalous; A003 is normal but neighbors are all anomalous
        geo = pa.table({
            "primary_key": ["A001", "A002", "A003"],
            "is_anomaly": [True, True, False],
        })
        nav._storage.read_geometry.return_value = geo

        # Phase 1: anomaly A001's neighbors include normal A003
        # Phase 2: A003's neighbors are anomalous A001, A002
        def fake_similar(pk, pat_id, top_n=10):
            if pk == "A001":
                return [("A003", 0.1), ("A002", 0.2)]
            if pk == "A003":
                return [("A001", 0.1), ("A002", 0.2)]
            return [("A001", 0.1), ("A003", 0.2)]

        nav.find_similar_entities = MagicMock(side_effect=fake_similar)

        with patch("random.sample", side_effect=lambda lst, n: lst[:n]):
            result = nav.detect_neighbor_contamination(
                "pat_a", k=2, sample_size=5,
                contamination_threshold=0.5,
            )

        # A003 (normal) should be found — its neighbors are all anomalous
        found_keys = {r["target_key"] for r in result}
        assert "A003" in found_keys
        assert result[0]["contamination_rate"] == 1.0


class TestDetectTrajectoryAnomaly:
    """Tests for detect_trajectory_anomaly."""

    def test_raises_on_event_pattern(self):
        """Event patterns should raise ValueError."""
        nav = _make_navigator()

        fake_pattern = MagicMock()
        fake_pattern.pattern_type = "event"

        fake_sphere = MagicMock()
        fake_sphere.patterns = {"evt_pat": fake_pattern}
        nav._storage.read_sphere.return_value = fake_sphere

        with pytest.raises(ValueError, match="event patterns have no temporal history"):
            nav.detect_trajectory_anomaly("evt_pat")

    def test_excludes_linear_drift(self):
        """Entities with linear_drift trajectory shape are excluded from results."""
        nav = _make_navigator()

        fake_pattern = MagicMock()
        fake_pattern.pattern_type = "anchor"
        fake_pattern.relations = [MagicMock(line_id="dim_a")]
        fake_pattern.sigma_diag = np.array([1.0], dtype=np.float32)
        fake_pattern.mu = np.array([0.0], dtype=np.float32)

        fake_sphere = MagicMock()
        fake_sphere.patterns = {"pat_a": fake_pattern}
        nav._storage.read_sphere.return_value = fake_sphere

        # Temporal data: monotonically increasing → linear_drift
        shapes = [[float(i) * 0.1] for i in range(10)]
        ts_base = 1704067200000000  # 2024-01-01 in μs
        temporal = pa.table({
            "primary_key": ["E001"] * 10,
            "shape_snapshot": shapes,
            "timestamp": pa.array(
                [ts_base + i * 86400_000_000 * 30 for i in range(10)],
                type=pa.timestamp("us", tz="UTC"),
            ),
        })

        def fake_temporal_batched(pattern_id, **kwargs):
            yield temporal.to_batches()[0]

        nav._storage.read_temporal_batched = MagicMock(side_effect=fake_temporal_batched)

        result = nav.detect_trajectory_anomaly("pat_a")
        # linear_drift is not an interesting shape → excluded
        assert len(result) == 0


class TestDetectSegmentShift:
    """Tests for detect_segment_shift."""

    def test_filters_by_min_shift_ratio(self):
        """Only segments meeting the shift ratio threshold are returned."""
        nav = _make_navigator()
        nav._manifest.pattern_version.return_value = 1
        nav._manifest.line_version.return_value = 1

        fake_pattern = MagicMock()
        fake_pattern.pattern_type = "anchor"
        fake_pattern.entity_line_id = "accounts"

        fake_col = MagicMock()
        fake_col.name = "region"
        fake_col.type = "string"

        fake_line = MagicMock()
        fake_line.columns = [fake_col]

        fake_sphere = MagicMock()
        fake_sphere.patterns = {"pat_a": fake_pattern}
        fake_sphere.entity_line.return_value = "accounts"
        fake_sphere.lines = {"accounts": fake_line}
        nav._storage.read_sphere.return_value = fake_sphere

        # Geometry: 100 entities, 10 anomalous (10% population rate)
        pk_list = [f"A{i:04d}" for i in range(100)]
        is_anom = [i < 10 for i in range(100)]  # first 10 are anomalous
        geo = pa.table({
            "primary_key": pk_list,
            "is_anomaly": is_anom,
        })
        nav._storage.read_geometry.return_value = geo

        # Regime change: no data
        nav.π12_attract_regime_change = MagicMock(return_value=[])

        # Points: first 20 entities are "region=EAST", rest are "region=WEST"
        regions = ["EAST" if i < 20 else "WEST" for i in range(100)]
        pts = pa.table({
            "primary_key": pk_list,
            "region": regions,
        })
        nav._storage.read_points.return_value = pts

        result = nav.detect_segment_shift(
            "pat_a", min_shift_ratio=2.0,
        )

        # EAST: 10/20 anomalous = 50%, shift_ratio = 50%/10% = 5.0 (included)
        # WEST: 0/80 anomalous = 0%, shift_ratio = 0 (excluded)
        assert len(result) == 1
        assert result[0]["segment_value"] == "EAST"
        assert result[0]["shift_ratio"] >= 2.0

    def test_skips_high_cardinality(self):
        """Columns with too many distinct values are skipped."""
        nav = _make_navigator()
        nav._manifest.pattern_version.return_value = 1
        nav._manifest.line_version.return_value = 1

        fake_pattern = MagicMock()
        fake_pattern.pattern_type = "anchor"
        fake_pattern.entity_line_id = "accounts"

        fake_col = MagicMock()
        fake_col.name = "account_name"
        fake_col.type = "string"

        fake_line = MagicMock()
        fake_line.columns = [fake_col]

        fake_sphere = MagicMock()
        fake_sphere.patterns = {"pat_a": fake_pattern}
        fake_sphere.entity_line.return_value = "accounts"
        fake_sphere.lines = {"accounts": fake_line}
        nav._storage.read_sphere.return_value = fake_sphere

        # Geometry: 100 entities, 10 anomalous
        pk_list = [f"A{i:04d}" for i in range(100)]
        is_anom = [i < 10 for i in range(100)]
        geo = pa.table({
            "primary_key": pk_list,
            "is_anomaly": is_anom,
        })
        nav._storage.read_geometry.return_value = geo

        nav.π12_attract_regime_change = MagicMock(return_value=[])

        # 100 distinct names → exceeds max_cardinality=50
        pts = pa.table({
            "primary_key": pk_list,
            "account_name": [f"Name_{i}" for i in range(100)],
        })
        nav._storage.read_points.return_value = pts

        result = nav.detect_segment_shift(
            "pat_a", max_cardinality=50,
        )

        # High cardinality column is skipped entirely
        assert result == []
