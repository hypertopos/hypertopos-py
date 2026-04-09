# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for geometry stats cache — write_geometry_stats / read_geometry_stats."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_delta_norms(n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=3.0, size=n).astype(np.float64)


# ---------------------------------------------------------------------------
# write_geometry_stats / read_geometry_stats round-trip
# ---------------------------------------------------------------------------


class TestGeometryStatsRoundTrip:
    def test_write_then_read_returns_stats(self, tmp_path: Path) -> None:
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        delta_norms = _make_delta_norms(200)
        theta_norm = 5.0

        writer.write_geometry_stats("test_pattern", 1, delta_norms, theta_norm)

        stats = reader.read_geometry_stats("test_pattern", 1)
        assert stats is not None
        assert stats["pattern_id"] == "test_pattern"
        assert stats["version"] == 1
        assert abs(stats["theta_norm"] - theta_norm) < 1e-6
        assert stats["total_entities"] == 200
        expected_anomalies = int(np.sum(delta_norms >= theta_norm))
        assert stats["total_anomalies"] == expected_anomalies

    def test_percentiles_present_and_ordered(self, tmp_path: Path) -> None:
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        delta_norms = _make_delta_norms(500)

        writer.write_geometry_stats("p", 1, delta_norms, theta_norm=3.0)

        stats = reader.read_geometry_stats("p", 1)
        assert stats is not None
        pcts = stats["percentiles"]
        assert set(pcts.keys()) == {"p50", "p75", "p90", "p95", "p99", "max"}
        assert (
            pcts["p50"] <= pcts["p75"] <= pcts["p90"] <= pcts["p95"] <= pcts["p99"] <= pcts["max"]
        )  # noqa: E501
        assert all(v >= 0.0 for v in pcts.values())

    def test_percentile_values_match_numpy(self, tmp_path: Path) -> None:
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        delta_norms = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        writer.write_geometry_stats("exact_p", 1, delta_norms, theta_norm=5.0)

        stats = reader.read_geometry_stats("exact_p", 1)
        assert stats is not None
        pcts = stats["percentiles"]
        assert abs(pcts["p50"] - float(np.percentile(delta_norms, 50))) < 1e-4
        assert abs(pcts["max"] - float(np.max(delta_norms))) < 1e-4

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        reader = GDSReader(str(tmp_path))
        result = reader.read_geometry_stats("nonexistent_pattern", 99)
        assert result is None

    def test_versioned_files_are_independent(self, tmp_path: Path) -> None:
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        norms_v1 = np.array([1.0, 2.0, 3.0])
        norms_v2 = np.array([10.0, 20.0, 30.0])

        writer.write_geometry_stats("pat", 1, norms_v1, theta_norm=2.5)
        writer.write_geometry_stats("pat", 2, norms_v2, theta_norm=15.0)

        s1 = reader.read_geometry_stats("pat", 1)
        s2 = reader.read_geometry_stats("pat", 2)
        assert s1 is not None and s2 is not None
        assert s1["total_entities"] == 3
        assert s2["total_entities"] == 3
        assert s1["percentiles"]["max"] < s2["percentiles"]["max"]

    def test_zero_total_anomalies_when_theta_very_large(self, tmp_path: Path) -> None:
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        delta_norms = np.array([1.0, 2.0, 3.0])

        writer.write_geometry_stats("pat", 1, delta_norms, theta_norm=100.0)

        stats = reader.read_geometry_stats("pat", 1)
        assert stats is not None
        assert stats["total_anomalies"] == 0


# ---------------------------------------------------------------------------
# write_lance_geometry auto-writes stats
# ---------------------------------------------------------------------------


class TestWriteLanceGeometryAutoStats:
    def _make_geo_table(self, n: int = 10) -> pa.Table:
        edge_struct_type = pa.struct(
            [
                pa.field("line_id", pa.string()),
                pa.field("point_key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("direction", pa.string()),
            ]
        )
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        delta_norms = np.linspace(1.0, 10.0, n, dtype=np.float32)
        return pa.table(
            {
                "primary_key": [f"E-{i:03d}" for i in range(n)],
                "scale": pa.array([1] * n, type=pa.int32()),
                "delta": [[float(i), float(i)] for i in range(n)],
                "delta_norm": delta_norms.tolist(),
                "delta_rank_pct": np.linspace(0, 100, n, dtype=np.float32).tolist(),
                "is_anomaly": [bool(v > 5.0) for v in delta_norms],
                "edges": pa.array([[]] * n, type=pa.list_(edge_struct_type)),
                "last_refresh_at": [ts] * n,
                "updated_at": [ts] * n,
            }
        )

    def test_stats_not_auto_created_by_write_lance_geometry(self, tmp_path: Path) -> None:
        """write_lance_geometry does NOT auto-write stats (theta_norm unknown at write time).
        Callers who know theta_norm must call write_geometry_stats explicitly."""
        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        geo_dir = tmp_path / "geometry" / "pat" / "v=1"
        table = self._make_geo_table(10)

        writer.write_lance_geometry(table, geo_dir)

        stats = reader.read_geometry_stats("pat", 1)
        assert stats is None  # stats must be written explicitly with correct theta_norm

    def test_stats_not_auto_created_by_build_index_if_needed(self, tmp_path: Path) -> None:
        """build_index_if_needed does NOT auto-write stats (theta_norm unknown).
        Callers must call write_geometry_stats explicitly after the fact."""
        import lance as _lance

        writer = GDSWriter(str(tmp_path))
        reader = GDSReader(str(tmp_path))
        table = self._make_geo_table(10)

        # Write geometry directly (bypasses write_lance_geometry)
        lance_path = tmp_path / "geometry" / "bi_pat" / "v=1" / "data.lance"
        lance_path.parent.mkdir(parents=True, exist_ok=True)
        delta_col = table["delta"]
        list_size = len(delta_col[0].as_py())
        fixed_delta = delta_col.cast(pa.list_(pa.float32(), list_size))
        table2 = table.set_column(table.schema.get_field_index("delta"), "delta", fixed_delta)
        _lance.write_dataset(table2, str(lance_path))

        writer.build_index_if_needed("bi_pat", version=1)

        stats = reader.read_geometry_stats("bi_pat", 1)
        assert stats is None  # stats must be written explicitly with correct theta_norm


# ---------------------------------------------------------------------------
# anomaly_summary uses cache (no delta column in scan)
# ---------------------------------------------------------------------------


class TestAnomalySummaryUsesCache:
    def test_anomaly_summary_uses_cached_percentiles(self, sphere_path: str) -> None:
        """When stats cache exists, anomaly_summary uses cached percentile values."""
        from datetime import datetime

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.model.manifest import Contract, Manifest
        from hypertopos.navigation.navigator import GDSNavigator
        from hypertopos.storage.cache import GDSCache

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-test",
            agent_id="a-test",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-test", ["customer_pattern"]),
        )
        summary = nav.anomaly_summary("customer_pattern")

        # Verify the fixture cache was used (stats file must exist)
        stats = reader.read_geometry_stats("customer_pattern", 1)
        assert stats is not None, "Fixture must have geometry stats cache"

        assert summary["total_entities"] == stats["total_entities"]
        # total_anomalies is recounted from stored delta_norm (not from stats
        # cache) — verify it matches a live count from geometry.
        geo = reader.read_geometry("customer_pattern", 1, columns=["delta_norm"])
        theta_norm = float(np.linalg.norm(
            np.array(reader.read_sphere().patterns["customer_pattern"].theta, dtype=np.float32),
        ))
        expected = int((geo["delta_norm"].to_numpy().astype(np.float64) >= theta_norm).sum())
        assert summary["total_anomalies"] == expected
        assert (
            abs(summary["delta_norm_percentiles"]["p50"] - round(stats["percentiles"]["p50"], 4))
            < 1e-3
        )  # noqa: E501

    def test_anomaly_summary_skips_delta_column_when_cached(
        self, sphere_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When stats cache hits, read_geometry should NOT load the delta column."""
        from datetime import datetime

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.model.manifest import Contract, Manifest
        from hypertopos.navigation.navigator import GDSNavigator
        from hypertopos.storage.cache import GDSCache

        loaded_columns: list[list[str] | None] = []
        original_read = GDSReader.read_geometry

        def patched_read(self, *args, columns: list[str] | None = None, **kwargs):  # type: ignore[override]
            loaded_columns.append(columns)
            return original_read(self, *args, columns=columns, **kwargs)

        monkeypatch.setattr(GDSReader, "read_geometry", patched_read)

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-test2",
            agent_id="a-test2",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-test2", ["customer_pattern"]),
        )

        # Ensure cache file exists in fixture
        stats = reader.read_geometry_stats("customer_pattern", 1)
        if stats is None:
            pytest.skip("Fixture has no geometry stats cache — regenerate fixtures")

        nav.anomaly_summary("customer_pattern")

        # The cluster scan should use explicit column list (not None = all columns)
        # and must NOT include "delta" as a standalone column (it is included for
        # anomaly cluster building but only after the cache supplies percentiles)
        # Verify: every read_geometry call has explicit columns, none is a full scan
        assert loaded_columns, "read_geometry must have been called at least once"
        for cols in loaded_columns:
            # Must never do a full scan (columns=None) in the cache-hit path
            assert cols is not None, (
                "anomaly_summary should not do a full geometry scan when cache hits"
            )

    def test_anomaly_summary_falls_back_without_cache(
        self, sphere_path: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a stats cache, anomaly_summary falls back to the full scan path."""
        from datetime import datetime

        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.model.manifest import Contract, Manifest
        from hypertopos.navigation.navigator import GDSNavigator
        from hypertopos.storage.cache import GDSCache

        # Simulate missing cache by monkeypatching read_geometry_stats to return None
        monkeypatch.setattr(GDSReader, "read_geometry_stats", lambda self, *a, **kw: None)

        reader = GDSReader(base_path=sphere_path)
        cache = GDSCache()
        engine = GDSEngine(storage=reader, cache=cache)
        manifest = Manifest(
            manifest_id="m-fb",
            agent_id="a-fb",
            snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
            status="active",
            line_versions={"customers": 1, "products": 1},
            pattern_versions={"customer_pattern": 1},
        )
        nav = GDSNavigator(
            engine=engine,
            storage=reader,
            manifest=manifest,
            contract=Contract("m-fb", ["customer_pattern"]),
        )
        # Should not raise — falls back to full scan
        summary = nav.anomaly_summary("customer_pattern")
        assert summary["total_entities"] == 2
        assert "delta_norm_percentiles" in summary

    def test_fixture_has_geometry_stats(self, sphere_path: str) -> None:
        """After fixture regeneration, geometry_stats file must exist."""
        reader = GDSReader(base_path=sphere_path)
        stats = reader.read_geometry_stats("customer_pattern", 1)
        assert stats is not None, (
            "Fixture missing geometry stats — run: "
            ".venv/Scripts/python tests/fixtures/generate_fixtures.py"
        )
        assert stats["percentiles"]["p50"] > 0
        assert stats["total_entities"] == 2
        assert stats["total_anomalies"] == 1  # CUST-001 is anomalous


def test_write_geometry_stats_theta_zero_reports_zero_anomalies(tmp_path: Path) -> None:
    """write_geometry_stats must report 0 anomalies when theta_norm=0."""
    import json

    import numpy as np
    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(tmp_path)
    delta_norms = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    writer.write_geometry_stats("p", 1, delta_norms, theta_norm=0.0)

    stats_path = tmp_path / "_gds_meta" / "geometry_stats" / "p_v1.json"
    stats = json.loads(stats_path.read_text())

    assert stats["total_anomalies"] == 0, (
        f"theta_norm=0 must produce 0 anomalies; got {stats['total_anomalies']}"
    )
