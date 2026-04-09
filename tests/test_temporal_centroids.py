# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for temporal centroid cache written at build_temporal time."""

from __future__ import annotations

from datetime import datetime

import lance
import pyarrow as pa
from hypertopos.builder.builder import GDSBuilder
from hypertopos.storage.reader import GDSReader


class TestTemporalCentroids:
    """Verify that build_temporal writes per-window centroid cache."""

    @staticmethod
    def _make_events(n_days: int = 3) -> pa.Table:
        rows: list[dict] = []
        tx_id = 0
        accounts = ["A", "B", "C"]
        for day in range(1, n_days + 1):
            for _ in range(10):
                rows.append(
                    {
                        "primary_key": f"TX-{tx_id}",
                        "from_account": accounts[tx_id % 3],
                        "to_account": accounts[(tx_id + 1) % 3],
                        "amount": (tx_id + 1) * 10,
                        "timestamp": datetime(2024, 1, day),
                    }
                )
                tx_id += 1
        return pa.table({k: [r[k] for r in rows] for k in rows[0]})

    @staticmethod
    def _build_with_temporal(
        tmp_path,
        n_days: int = 3,
        time_window: str = "1d",
    ) -> tuple[GDSBuilder, dict[str, int]]:
        events = TestTemporalCentroids._make_events(n_days)
        accounts = pa.table({"primary_key": ["A", "B", "C"]})
        builder = GDSBuilder("test", str(tmp_path / "sphere"))
        builder.add_line(
            "transactions",
            events,
            key_col="primary_key",
            source_id="test",
            role="event",
        )
        builder.add_line(
            "accounts",
            accounts,
            key_col="primary_key",
            source_id="test",
            role="anchor",
        )
        builder.add_derived_dimension(
            "accounts",
            "transactions",
            "from_account",
            "count",
            None,
            "tx_count",
        )
        builder.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
        )
        builder.build()
        result = builder.build_temporal("timestamp", time_window)
        return builder, result

    def test_centroid_lance_exists(self, tmp_path):
        """Temporal centroid Lance dataset is created after build_temporal."""
        _builder, result = self._build_with_temporal(tmp_path, n_days=3)
        assert result["acct_pattern"] == 3

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        assert centroid_path.exists()

    def test_centroid_schema(self, tmp_path):
        """Centroid dataset has the expected columns and types."""
        self._build_with_temporal(tmp_path, n_days=3)

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        ds = lance.dataset(str(centroid_path))
        field_names = {f.name for f in ds.schema}
        expected = {
            "window_start",
            "window_end",
            "centroid",
            "entity_count",
            "anomaly_rate",
        }
        assert expected == field_names

    def test_centroid_row_count_matches_windows(self, tmp_path):
        """Number of centroid rows matches number of temporal windows."""
        _builder, result = self._build_with_temporal(
            tmp_path,
            n_days=5,
            time_window="1d",
        )
        n_slices = result["acct_pattern"]
        assert n_slices == 5

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        ds = lance.dataset(str(centroid_path))
        assert ds.count_rows() == n_slices

    def test_centroid_entity_count_positive(self, tmp_path):
        """Every centroid row has a positive entity_count."""
        self._build_with_temporal(tmp_path, n_days=3)

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        ds = lance.dataset(str(centroid_path))
        tbl = ds.to_table()
        counts = tbl["entity_count"].to_pylist()
        assert all(c > 0 for c in counts)

    def test_centroid_anomaly_rate_range(self, tmp_path):
        """Anomaly rate is in [0.0, 1.0] for every window."""
        self._build_with_temporal(tmp_path, n_days=3)

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        ds = lance.dataset(str(centroid_path))
        tbl = ds.to_table()
        rates = tbl["anomaly_rate"].to_pylist()
        assert all(0.0 <= r <= 1.0 for r in rates)

    def test_centroid_vector_length_matches_dimensions(self, tmp_path):
        """Centroid vector length matches pattern dimension count."""
        self._build_with_temporal(tmp_path, n_days=3)

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        ds = lance.dataset(str(centroid_path))
        tbl = ds.to_table()
        centroids = tbl["centroid"].to_pylist()

        # Pattern has 1 derived dim (tx_count) → D=1
        for c in centroids:
            assert len(c) == 1

    def test_window_timestamps_are_ordered(self, tmp_path):
        """Window start timestamps are strictly increasing."""
        self._build_with_temporal(tmp_path, n_days=5, time_window="1d")

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        ds = lance.dataset(str(centroid_path))
        tbl = ds.to_table()
        starts = tbl["window_start"].to_pylist()
        for i in range(1, len(starts)):
            assert starts[i] > starts[i - 1]

    def test_window_end_after_start(self, tmp_path):
        """Each window_end is after its window_start."""
        self._build_with_temporal(tmp_path, n_days=3)

        centroid_path = (
            tmp_path / "sphere" / "_gds_meta" / "temporal_centroids" / "acct_pattern.lance"
        )
        ds = lance.dataset(str(centroid_path))
        tbl = ds.to_table()
        starts = tbl["window_start"].to_pylist()
        ends = tbl["window_end"].to_pylist()
        for s, e in zip(starts, ends, strict=False):
            assert e > s


class TestReadTemporalCentroids:
    """Verify GDSReader.read_temporal_centroids returns the written cache."""

    def test_read_returns_list(self, tmp_path):
        """Reader returns a non-empty list of dicts for an existing cache."""
        TestTemporalCentroids._build_with_temporal(tmp_path, n_days=3)
        reader = GDSReader(str(tmp_path / "sphere"))
        centroids = reader.read_temporal_centroids("acct_pattern")
        assert centroids is not None
        assert isinstance(centroids, list)
        assert len(centroids) > 0

    def test_read_dict_keys(self, tmp_path):
        """Each centroid dict has all required keys."""
        TestTemporalCentroids._build_with_temporal(tmp_path, n_days=3)
        reader = GDSReader(str(tmp_path / "sphere"))
        centroids = reader.read_temporal_centroids("acct_pattern")
        assert centroids is not None
        for c in centroids:
            assert "window_start" in c
            assert "window_end" in c
            assert "centroid" in c
            assert "entity_count" in c
            assert "anomaly_rate" in c

    def test_read_row_count_matches_windows(self, tmp_path):
        """Number of returned dicts matches the number of temporal windows built."""
        _builder, result = TestTemporalCentroids._build_with_temporal(
            tmp_path,
            n_days=4,
            time_window="1d",
        )
        reader = GDSReader(str(tmp_path / "sphere"))
        centroids = reader.read_temporal_centroids("acct_pattern")
        assert centroids is not None
        assert len(centroids) == result["acct_pattern"]

    def test_read_missing_returns_none(self, tmp_path):
        """Returns None for a pattern with no centroid cache."""
        reader = GDSReader(str(tmp_path))
        assert reader.read_temporal_centroids("nonexistent") is None


class TestPi11UsesCache:
    """Verify pi11 returns results from centroid cache with 'cached' field."""

    def test_pi11_uses_cache(self, tmp_path):
        """pi11 returns results from cache with 'cached' field."""
        from hypertopos.sphere import HyperSphere

        TestTemporalCentroids._build_with_temporal(
            tmp_path,
            n_days=4,
            time_window="1d",
        )
        hs = HyperSphere.open(str(tmp_path / "sphere"))
        session = hs.session("test-agent")
        nav = session.navigator()

        result = nav.π11_attract_population_compare(
            "acct_pattern",
            window_a_from="2024-01-01",
            window_a_to="2024-01-03",
            window_b_from="2024-01-03",
            window_b_to="2024-01-05",
        )
        assert result.get("cached") is True
        assert "centroid_shift" in result
        assert result["centroid_shift"] is not None
        assert "top_changed_dimensions" in result
        assert "interpretation" in result
        assert result["window_a"]["entry_count"] > 0
        assert result["window_b"]["entry_count"] > 0

    def test_pi11_cache_empty_window(self, tmp_path):
        """pi11 returns warning when one window has no cached data."""
        from hypertopos.sphere import HyperSphere

        TestTemporalCentroids._build_with_temporal(
            tmp_path,
            n_days=3,
            time_window="1d",
        )
        hs = HyperSphere.open(str(tmp_path / "sphere"))
        session = hs.session("test-agent")
        nav = session.navigator()

        result = nav.π11_attract_population_compare(
            "acct_pattern",
            window_a_from="2024-01-01",
            window_a_to="2024-01-02",
            window_b_from="2025-06-01",
            window_b_to="2025-07-01",
        )
        # Cache had no matching windows → falls through to full scan
        # Full scan also finds no data → returns warning
        assert result["centroid_shift"] is None or result.get("warning")


class TestPi12UsesCache:
    """Verify pi12 returns results from centroid cache."""

    def test_pi12_uses_cache(self, tmp_path):
        """pi12 returns list[dict] with expected keys when centroid cache is available."""
        from hypertopos.sphere import HyperSphere

        TestTemporalCentroids._build_with_temporal(
            tmp_path,
            n_days=5,
            time_window="1d",
        )
        hs = HyperSphere.open(str(tmp_path / "sphere"))
        session = hs.session("test-agent")
        nav = session.navigator()

        result = nav.π12_attract_regime_change("acct_pattern", n_regimes=3)
        assert isinstance(result, list)
        for item in result:
            assert "timestamp" in item
            assert "magnitude" in item
            assert "top_changed_dimensions" in item
            assert "description" in item

    def test_pi12_cache_respects_time_filter(self, tmp_path):
        """pi12 filters by timestamp_from / timestamp_to when cache is used."""
        from hypertopos.sphere import HyperSphere

        TestTemporalCentroids._build_with_temporal(
            tmp_path,
            n_days=5,
            time_window="1d",
        )
        hs = HyperSphere.open(str(tmp_path / "sphere"))
        session = hs.session("test-agent")
        nav = session.navigator()

        # Range covering only first 2 days — fewer windows than full scan
        result_filtered = nav.π12_attract_regime_change(
            "acct_pattern",
            timestamp_from="2024-01-01",
            timestamp_to="2024-01-03",
            n_regimes=3,
        )
        result_full = nav.π12_attract_regime_change("acct_pattern", n_regimes=3)
        # Filtered result must be a subset: same type, no extra entries
        assert isinstance(result_filtered, list)
        assert len(result_filtered) <= len(result_full)

    def test_pi12_cache_empty_range_returns_empty(self, tmp_path):
        """pi12 returns empty list when time filter matches fewer than 2 windows."""
        from hypertopos.sphere import HyperSphere

        TestTemporalCentroids._build_with_temporal(
            tmp_path,
            n_days=3,
            time_window="1d",
        )
        hs = HyperSphere.open(str(tmp_path / "sphere"))
        session = hs.session("test-agent")
        nav = session.navigator()

        result = nav.π12_attract_regime_change(
            "acct_pattern",
            timestamp_from="2025-06-01",
            timestamp_to="2025-07-01",
            n_regimes=3,
        )
        assert result == []
