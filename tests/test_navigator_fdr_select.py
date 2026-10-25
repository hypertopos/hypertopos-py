# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for fdr_alpha and select params wired into navigator Tier A methods."""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.engine import fdr as fdr_module
from hypertopos.engine.fdr import benjamini_hochberg, parametric_p_values_chi2
from hypertopos.model.manifest import Contract, Manifest
from hypertopos.model.objects import Polygon
from hypertopos.model.sphere import (
    Alias,
    AliasFilter,
    CuttingPlane,
    DerivedPattern,
    Pattern,
    RelationDef,
    Sphere,
)
from hypertopos.navigation.navigator import GDSNavigator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DT = datetime(2024, 1, 1, tzinfo=UTC)

_PATTERN = Pattern(
    pattern_id="test_pattern",
    entity_type="test",
    pattern_type="anchor",
    relations=[
        RelationDef(line_id="line_a", direction="in", required=True),
        RelationDef(line_id="line_b", direction="in", required=True),
    ],
    mu=np.zeros(2, dtype=np.float32),
    sigma_diag=np.ones(2, dtype=np.float32),
    theta=np.array([0.0, 3.0], dtype=np.float32),
    population_size=20,
    computed_at=_DT,
    version=1,
    status="production",
)

_MANIFEST = Manifest(
    manifest_id="m1",
    agent_id="test",
    snapshot_time=_DT,
    status="active",
    line_versions={"line_a": 1, "line_b": 1},
    pattern_versions={"test_pattern": 1},
)

_CONTRACT = Contract(manifest_id="m1", pattern_ids=["test_pattern"])


def _make_geometry_table(n: int = 20) -> pa.Table:
    """Create a geometry table with n entities, linearly spread norms."""
    keys = [f"E-{i:03d}" for i in range(n)]
    deltas = []
    norms = []
    for i in range(n):
        # Spread norms from 4.0 (highest, anomalous) down to just above threshold
        d = np.array([3.0 + float(i) / n, 3.0 + float(i) / n], dtype=np.float32)
        deltas.append(d.tolist())
        norms.append(float(np.linalg.norm(d)))
    return pa.table(
        {
            "primary_key": keys,
            "scale": [1] * n,
            "delta": deltas,
            "delta_norm": norms,
            "is_anomaly": [True] * n,
            "delta_rank_pct": pa.array([float(i) / n * 100 for i in range(n)], type=pa.float64()),
            "last_refresh_at": [_DT] * n,
            "updated_at": [_DT] * n,
        }
    )


# ---------------------------------------------------------------------------
# Mock storage for pi5
# ---------------------------------------------------------------------------


class _MockStoragePi5:
    """Mock storage that returns geometry for pi5 tests."""

    def __init__(self, n: int = 20):
        self._table = _make_geometry_table(n)
        self._n = n

    def read_sphere(self):
        sphere = Sphere("s", "s", ".")
        sphere.patterns["test_pattern"] = _PATTERN
        return sphere

    def count_geometry_rows(self, *a, **kw):
        return 0

    def read_geometry(
        self,
        pattern_id,
        version,
        primary_key=None,
        filters=None,
        point_keys=None,
        columns=None,
        filter=None,
    ):
        table = self._table
        if filter is not None:
            if "primary_key IN" in str(filter):
                keys = re.findall(r"'([^']*)'", str(filter))
                import pyarrow.compute as _pc

                table = table.filter(
                    _pc.is_in(table["primary_key"], pa.array(keys)),
                )
            else:
                threshold = float(filter.split(">=")[1].strip())
                import pyarrow.compute as _pc

                table = table.filter(
                    _pc.greater_equal(table["delta_norm"], threshold),
                )
        if point_keys is not None:
            import pyarrow.compute as _pc

            mask = _pc.is_in(table["primary_key"], pa.array(point_keys))
            table = table.filter(mask)
        if columns is not None:
            available = set(table.schema.names)
            columns = [c for c in columns if c in available]
            if columns:
                table = table.select(columns)
        return table

    def read_geometry_stats(self, *a, **kw):
        return None


# ---------------------------------------------------------------------------
# Mock storage for pi6
# ---------------------------------------------------------------------------


class _MockStoragePi6(_MockStoragePi5):
    """Add alias to the sphere for pi6."""

    def read_sphere(self):
        sphere = super().read_sphere()
        cp = CuttingPlane(normal=[1.0, 0.0], bias=0.5)
        alias_filter = AliasFilter(
            include_relations=["line_a"],
            cutting_plane=cp,
        )
        derived = DerivedPattern(
            mu=np.zeros(2, dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.array([3.0, 3.0], dtype=np.float32),
            population_size=10,
            computed_at=_DT,
        )
        sphere.aliases["test_alias"] = Alias(
            alias_id="test_alias",
            base_pattern_id="test_pattern",
            filter=alias_filter,
            derived_pattern=derived,
            version=1,
            status="production",
        )
        return sphere


# ---------------------------------------------------------------------------
# Mock storage for pi7
# ---------------------------------------------------------------------------


class _MockStoragePi7(_MockStoragePi5):
    """Returns patterns with edge_max so pi7 uses the continuous path."""

    def read_sphere(self):
        sphere = Sphere("s", "s", ".")
        pat = Pattern(
            pattern_id="test_pattern",
            entity_type="test",
            pattern_type="anchor",
            relations=[
                RelationDef(line_id="line_a", direction="in", required=True),
                RelationDef(line_id="line_b", direction="in", required=True),
            ],
            mu=np.array([0.5, 0.5], dtype=np.float32),
            sigma_diag=np.ones(2, dtype=np.float32),
            theta=np.array([0.0, 3.0], dtype=np.float32),
            population_size=20,
            computed_at=_DT,
            version=1,
            status="production",
            edge_max=np.array([10.0, 10.0], dtype=np.float32),
        )
        sphere.patterns["test_pattern"] = pat
        return sphere


# ---------------------------------------------------------------------------
# Mock storage for pi9
# ---------------------------------------------------------------------------


class _MockStoragePi9(_MockStoragePi5):
    """Returns temporal data for pi9 drift tests."""

    def read_temporal_batched(self, pattern_id, timestamp_from=None, timestamp_to=None, keys=None):
        """Yield temporal batches with shape_snapshot for drift computation."""
        n = self._n
        all_keys = [f"E-{i:03d}" for i in range(n)]
        if keys is not None:
            all_keys = [k for k in all_keys if k in keys]

        # For each entity, 3 temporal slices with increasing displacement
        rows_pk = []
        rows_shape = []
        rows_ts = []
        for i, pk in enumerate(all_keys):
            for s in range(3):
                rows_pk.append(pk)
                # shape = mu + delta * sigma => shape values increase over time
                shape = [0.5 + float(i * s) / (n * 3), 0.5 + float(i * s) / (n * 3)]
                rows_shape.append(shape)
                ts = datetime(2024, 1 + s, 1, tzinfo=UTC)
                rows_ts.append(ts)

        batch = pa.RecordBatch.from_pydict(
            {
                "primary_key": rows_pk,
                "shape_snapshot": rows_shape,
                "timestamp": pa.array(rows_ts, type=pa.timestamp("us", tz="UTC")),
            }
        )
        yield batch

    def _apply_temporal_filters(self, table, filters):
        return table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = MagicMock()
    eng.geometry_to_polygons = MagicMock(return_value=[])
    return eng


@pytest.fixture
def nav_pi5(engine):
    storage = _MockStoragePi5()
    return GDSNavigator(engine, storage, _MANIFEST, _CONTRACT)


@pytest.fixture
def nav_pi6(engine):
    storage = _MockStoragePi6()
    return GDSNavigator(engine, storage, _MANIFEST, _CONTRACT)


@pytest.fixture
def nav_pi7(engine):
    storage = _MockStoragePi7()
    return GDSNavigator(engine, storage, _MANIFEST, _CONTRACT)


@pytest.fixture
def nav_pi9(engine):
    storage = _MockStoragePi9()
    return GDSNavigator(engine, storage, _MANIFEST, _CONTRACT)


# ======================================================================
# π5 attract_anomaly
# ======================================================================


class TestPi5FdrSelect:
    def test_pi5_fdr_alpha_none_unchanged(self, nav_pi5):
        """Without fdr_alpha, output is identical to baseline."""
        base, total, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=10,
        )
        with_none, total2, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=10,
            fdr_alpha=None,
        )
        assert len(base) == len(with_none)
        assert total == total2

    def test_pi5_fdr_alpha_filters(self, nav_pi5):
        """With fdr_alpha=0.05, result is a subset with q_value attrs."""
        base, total_base, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=20,
        )
        filtered, total_fdr, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=20,
            fdr_alpha=0.05,
        )
        assert len(filtered) <= len(base)
        # total_found reflects pre-FDR count
        assert total_fdr == total_base
        # All surviving polygons must have q_value set
        for p in filtered:
            assert hasattr(p, "q_value")
            assert p.q_value <= 0.05  # type: ignore[attr-defined]

    def test_pi5_select_diverse(self, nav_pi5):
        """With select='diverse', result has representativeness attrs."""
        results, _, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=5,
            select="diverse",
        )
        if len(results) > 0:
            for p in results:
                assert hasattr(p, "representativeness")
                assert p.representativeness >= 0  # type: ignore[attr-defined]

    def test_pi5_compose_fdr_and_diverse(self, nav_pi5):
        """FDR + diverse together work without error."""
        results, _, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=10,
            fdr_alpha=0.5,
            select="diverse",
        )
        # Should be a valid list (may be empty if FDR filters everything)
        assert isinstance(results, list)

    def test_pi5_select_invalid_raises(self, nav_pi5):
        """select='quantum' raises ValueError."""
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi5.π5_attract_anomaly(
                "test_pattern",
                top_n=5,
                select="quantum",
            )

    def test_pi5_fdr_method_storey_runs(self, nav_pi5):
        """fdr_method='storey' returns results without raising."""
        polys, total, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=5,
            fdr_alpha=0.1,
            fdr_method="storey",
        )
        assert total >= 0
        assert isinstance(polys, list)

    def test_pi5_fdr_method_invalid_raises(self, nav_pi5):
        with pytest.raises(ValueError, match="fdr_method must be"):
            nav_pi5.π5_attract_anomaly(
                "test_pattern",
                top_n=5,
                fdr_method="bogus",
            )

    def test_pi5_rank_by_invalid_raises(self, nav_pi5):
        with pytest.raises(ValueError, match="rank_by must be"):
            nav_pi5.π5_attract_anomaly(
                "test_pattern",
                top_n=5,
                rank_by="bogus",
            )

    def test_pi5_rank_by_min_q_per_dim_requires_fdr_alpha(self, nav_pi5):
        with pytest.raises(ValueError, match="requires fdr_alpha"):
            nav_pi5.π5_attract_anomaly(
                "test_pattern",
                top_n=5,
                rank_by="min_q_per_dim",
                fdr_alpha=None,
                fdr_axis="per_dim",
            )

    def test_pi5_rank_by_min_q_per_dim_requires_per_dim_axis(self, nav_pi5):
        with pytest.raises(ValueError, match="fdr_axis"):
            nav_pi5.π5_attract_anomaly(
                "test_pattern",
                top_n=5,
                rank_by="min_q_per_dim",
                fdr_alpha=0.05,
                fdr_axis="entity",
            )

    def test_pi5_rank_by_min_q_per_dim_incompatible_with_diverse(self, nav_pi5):
        with pytest.raises(ValueError, match="select='diverse'"):
            nav_pi5.π5_attract_anomaly(
                "test_pattern",
                top_n=5,
                rank_by="min_q_per_dim",
                fdr_alpha=0.05,
                fdr_axis="per_dim",
                select="diverse",
            )


# ======================================================================
# π6 attract_boundary
# ======================================================================


class TestPi6FdrSelect:
    def test_pi6_fdr_alpha_none_unchanged(self, nav_pi6):
        """Without fdr_alpha, output is identical to baseline."""
        base = nav_pi6.π6_attract_boundary(
            "test_alias",
            "test_pattern",
            top_n=10,
        )
        with_none = nav_pi6.π6_attract_boundary(
            "test_alias",
            "test_pattern",
            top_n=10,
            fdr_alpha=None,
        )
        assert len(base) == len(with_none)

    def test_pi6_fdr_alpha_filters(self, nav_pi6):
        """With fdr_alpha=0.05, result is a subset with q_value attrs."""
        base = nav_pi6.π6_attract_boundary(
            "test_alias",
            "test_pattern",
            top_n=20,
        )
        filtered = nav_pi6.π6_attract_boundary(
            "test_alias",
            "test_pattern",
            top_n=20,
            fdr_alpha=0.05,
        )
        assert len(filtered) <= len(base)
        for poly, _dist in filtered:
            assert hasattr(poly, "q_value")
            assert poly.q_value <= 0.05  # type: ignore[attr-defined]

    def test_pi6_select_diverse(self, nav_pi6):
        """With select='diverse', result has representativeness attrs."""
        results = nav_pi6.π6_attract_boundary(
            "test_alias",
            "test_pattern",
            top_n=5,
            select="diverse",
        )
        if len(results) > 0:
            for poly, _ in results:
                assert hasattr(poly, "representativeness")
                assert poly.representativeness >= 0  # type: ignore[attr-defined]

    def test_pi6_compose_fdr_and_diverse(self, nav_pi6):
        """FDR + diverse together work without error."""
        results = nav_pi6.π6_attract_boundary(
            "test_alias",
            "test_pattern",
            top_n=10,
            fdr_alpha=0.5,
            select="diverse",
        )
        assert isinstance(results, list)

    def test_pi6_select_invalid_raises(self, nav_pi6):
        """select='quantum' raises ValueError."""
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi6.π6_attract_boundary(
                "test_alias",
                "test_pattern",
                top_n=5,
                select="quantum",
            )

    def test_pi6_fdr_method_storey_runs(self, nav_pi6):
        """fdr_method='storey' returns results without raising."""
        result = nav_pi6.π6_attract_boundary(
            "test_alias",
            "test_pattern",
            top_n=5,
            fdr_alpha=0.1,
            fdr_method="storey",
        )
        assert result is not None

    def test_pi6_fdr_method_invalid_raises(self, nav_pi6):
        with pytest.raises(ValueError, match="fdr_method must be"):
            nav_pi6.π6_attract_boundary(
                "test_alias",
                "test_pattern",
                top_n=5,
                fdr_method="bogus",
            )


# ======================================================================
# π7 attract_hub
# ======================================================================


class TestPi7Select:
    def test_pi7_select_diverse(self, nav_pi7):
        """With select='diverse', result is reordered."""
        results = nav_pi7.π7_attract_hub(
            "test_pattern",
            top_n=5,
            select="diverse",
        )
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_pi7_select_invalid_raises(self, nav_pi7):
        """select='quantum' raises ValueError."""
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi7.π7_attract_hub(
                "test_pattern",
                top_n=5,
                select="quantum",
            )


# ======================================================================
# π9 attract_drift
# ======================================================================


class TestPi9Select:
    def test_pi9_select_diverse(self, nav_pi9):
        """With select='diverse', result has representativeness keys."""
        results = nav_pi9.π9_attract_drift(
            "test_pattern",
            top_n=5,
            select="diverse",
        )
        if len(results) > 0:
            for row in results:
                assert "representativeness" in row
                assert row["representativeness"] >= 0

    def test_pi9_select_invalid_raises(self, nav_pi9):
        """select='quantum' raises ValueError."""
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi9.π9_attract_drift(
                "test_pattern",
                top_n=5,
                select="quantum",
            )


# ======================================================================
# Cross-method: fdr_method validation
# ======================================================================


class TestFdrMethodValidation:
    def test_fdr_method_invalid_raises_pi5(self, nav_pi5):
        with pytest.raises(ValueError, match="fdr_method must be"):
            nav_pi5.π5_attract_anomaly("test_pattern", fdr_method="bogus")

    def test_fdr_method_invalid_raises_pi6(self, nav_pi6):
        with pytest.raises(ValueError, match="fdr_method must be"):
            nav_pi6.π6_attract_boundary(
                "test_alias",
                "test_pattern",
                fdr_method="bogus",
            )

    def test_storey_returns_at_least_as_many_as_bh_pi5(self, nav_pi5):
        """Storey scales q-values by pi0 <= 1, so discoveries >= BH at same alpha."""
        polys_bh, _, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=100,
            fdr_alpha=0.1,
            fdr_method="bh",
        )
        polys_st, _, _, _ = nav_pi5.π5_attract_anomaly(
            "test_pattern",
            top_n=100,
            fdr_alpha=0.1,
            fdr_method="storey",
        )
        assert len(polys_st) >= len(polys_bh)


# ======================================================================
# FDR runs only on χ² model p-values; rank-derived p-values are gone.
# Rank-derived p-values are uniform by construction — they encode an
# entity's position in the result set, not evidence against the null — so
# BH on them selected a fixed fraction of whatever result set was requested.
# ======================================================================

_FDR_PARAMS = ("fdr_alpha", "fdr_method", "p_value_method")


class _MockStorageNoise(_MockStoragePi5):
    """Pure-noise population: δ ~ N(0, 1)^2 for every entity, population_size = n."""

    def __init__(self, n: int = 5000, seed: int = 7):
        rng = np.random.default_rng(seed)
        deltas = rng.standard_normal((n, 2)).astype(np.float32)
        norms = np.linalg.norm(deltas, axis=1).astype(np.float64)
        self._n = n
        self._table = pa.table(
            {
                "primary_key": [f"N-{i:05d}" for i in range(n)],
                "scale": [1] * n,
                "delta": deltas.tolist(),
                "delta_norm": norms,
                "is_anomaly": (norms >= 3.0).tolist(),
                "delta_rank_pct": pa.array([None] * n, type=pa.float64()),
                "last_refresh_at": [_DT] * n,
                "updated_at": [_DT] * n,
            }
        )

    def read_sphere(self):
        import dataclasses

        sphere = Sphere("s", "s", ".")
        sphere.patterns["test_pattern"] = dataclasses.replace(_PATTERN, population_size=self._n)
        return sphere


class TestFdrOverTestedPopulation:
    def test_pi5_top_n_on_pure_noise_keeps_no_survivors(self):
        """BH must count every tested entity, not only the returned top_n."""
        from hypertopos.engine.geometry import GDSEngine
        from hypertopos.storage.cache import GDSCache

        storage = _MockStorageNoise()
        nav = GDSNavigator(
            GDSEngine(storage=storage, cache=GDSCache()), storage, _MANIFEST, _CONTRACT
        )
        results, *_ = nav.π5_attract_anomaly("test_pattern", top_n=10, fdr_alpha=0.05)
        assert len(results) <= 1

    def test_subset_q_values_match_population_bh(self):
        from hypertopos.engine.fdr import benjamini_hochberg, parametric_p_values_chi2

        rng = np.random.default_rng(3)
        null = np.linalg.norm(rng.standard_normal((9800, 4)), axis=1)
        signal = np.linalg.norm(rng.standard_normal((200, 4)) + 3.0, axis=1)
        norms = np.sort(np.concatenate([null, signal]))[::-1]
        p = parametric_p_values_chi2(norms, df=4)
        full_rej, full_q = benjamini_hochberg(p, 0.05)
        top = 300
        sub_rej, sub_q = benjamini_hochberg(p[:top], 0.05, n_tests=p.size)
        assert np.array_equal(sub_rej, full_rej[:top])
        assert np.all(sub_q >= full_q[:top] - 1e-12)
        # Offset page: ranks continue after the skipped prefix.
        page_rej, _ = benjamini_hochberg(p[100:top], 0.05, n_tests=p.size, rank_offset=100)
        assert np.array_equal(page_rej, full_rej[100:top])


class TestRankFdrRemoved:
    def test_rank_p_value_method_rejected(self, nav_pi5):
        with pytest.raises(ValueError, match="p_value_method must be 'chi2'"):
            nav_pi5.π5_attract_anomaly("test_pattern", fdr_alpha=0.05, p_value_method="rank")

    def test_pi6_rank_p_value_method_rejected(self, nav_pi6):
        with pytest.raises(ValueError, match="p_value_method must be 'chi2'"):
            nav_pi6.π6_attract_boundary(
                "test_alias", "test_pattern", fdr_alpha=0.05, p_value_method="rank"
            )

    def test_pi5_pi6_default_p_value_method_is_chi2(self):
        for fn in (GDSNavigator.π5_attract_anomaly, GDSNavigator.π6_attract_boundary):
            assert inspect.signature(fn).parameters["p_value_method"].default == "chi2"

    @pytest.mark.parametrize("method_name", ["π7_attract_hub", "π7_attract_hub_and_stats"])
    def test_pi7_has_no_fdr_parameters(self, nav_pi7, method_name):
        params = inspect.signature(getattr(GDSNavigator, method_name)).parameters
        assert not set(_FDR_PARAMS) & set(params)
        with pytest.raises(TypeError):
            getattr(nav_pi7, method_name)("test_pattern", top_n=5, fdr_alpha=0.05)

    def test_pi9_has_no_fdr_parameters(self, nav_pi9):
        params = inspect.signature(GDSNavigator.π9_attract_drift).parameters
        assert not set(_FDR_PARAMS) & set(params)
        with pytest.raises(TypeError):
            nav_pi9.π9_attract_drift("test_pattern", top_n=5, fdr_alpha=0.05)

    def test_pi9_results_carry_no_q_value(self, nav_pi9):
        for row in nav_pi9.π9_attract_drift("test_pattern", top_n=20):
            assert "q_value" not in row

    def test_rank_p_value_helper_removed(self):
        assert not hasattr(fdr_module, "empirical_p_values_from_rank")
        assert "empirical_p_values_from_rank" not in fdr_module.__all__

    def test_pi5_chi2_fdr_noise_rejection_rate(self):
        """Pure-noise δ ~ N(0, 1)^D: χ² FDR at α must reject at most ~α of
        entities, and the default selection path must use χ² p-values."""
        rng = np.random.default_rng(20261021)
        n, d, alpha = 5000, 8, 0.05
        deltas = rng.standard_normal((n, d))
        norms = np.linalg.norm(deltas, axis=1)
        # Population rank percentiles — the input rank-FDR used to consume.
        order = np.argsort(np.argsort(norms))
        polygons = [
            Polygon(
                primary_key=f"N-{i:05d}",
                pattern_id="test_pattern",
                pattern_ver=1,
                pattern_type="anchor",
                scale=1,
                delta=deltas[i],
                delta_norm=float(norms[i]),
                is_anomaly=False,
                edges=[],
                last_refresh_at=_DT,
                updated_at=_DT,
                delta_rank_pct=float(order[i]) / n * 100.0,
            )
            for i in range(n)
        ]
        survivors = GDSNavigator._apply_fdr_select_polygons(
            polygons,
            fdr_alpha=alpha,
            select="top_norm",
            top_n=n,
            pattern_df=d,
            n_tests=n,
        )
        assert len(survivors) / n <= 0.08

        _, expected_q = benjamini_hochberg(parametric_p_values_chi2(norms, df=d), alpha)
        got_q = np.array([p.q_value for p in polygons])  # type: ignore[attr-defined]
        np.testing.assert_allclose(got_q, expected_q)
