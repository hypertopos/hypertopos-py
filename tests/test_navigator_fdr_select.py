# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Selection, ranking and the removed entity-level FDR on navigator Tier A methods."""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.engine import fdr as fdr_module
from hypertopos.engine.fdr import benjamini_hochberg
from hypertopos.model.manifest import Contract, Manifest
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
            "score_rank_pct": pa.array([float(i) / n * 100 for i in range(n)], type=pa.float64()),
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
            elif str(filter) == "is_anomaly":
                import pyarrow.compute as _pc

                table = table.filter(table["is_anomaly"])
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
# π5 attract_anomaly — selection, ranking, removed entity FDR
# ======================================================================

_REMOVED_FDR_PARAMS = ("p_value_method", "fdr_axis")


class TestPi5Select:
    def test_pi5_select_diverse(self, nav_pi5):
        results, _, _, _ = nav_pi5.π5_attract_anomaly("test_pattern", top_n=5, select="diverse")
        assert results
        for p in results:
            assert p.representativeness >= 0  # type: ignore[attr-defined]

    def test_pi5_select_invalid_raises(self, nav_pi5):
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi5.π5_attract_anomaly("test_pattern", top_n=5, select="quantum")

    def test_pi5_fdr_method_invalid_raises(self, nav_pi5):
        with pytest.raises(ValueError, match="fdr_method must be"):
            nav_pi5.π5_attract_anomaly("test_pattern", top_n=5, fdr_method="bogus")

    def test_pi5_rank_by_invalid_raises(self, nav_pi5):
        with pytest.raises(ValueError, match="rank_by must be"):
            nav_pi5.π5_attract_anomaly("test_pattern", top_n=5, rank_by="bogus")

    def test_pi5_rank_by_min_q_per_dim_removed(self, nav_pi5):
        with pytest.raises(ValueError, match="rank_by must be"):
            nav_pi5.π5_attract_anomaly("test_pattern", top_n=5, rank_by="min_q_per_dim")

    def test_pi5_fdr_alpha_without_cell_resolution_raises(self, nav_pi5):
        with pytest.raises(ValueError, match="cell gate"):
            nav_pi5.π5_attract_anomaly("test_pattern", top_n=5, fdr_alpha=0.05)

    @pytest.mark.parametrize("name", _REMOVED_FDR_PARAMS)
    def test_pi5_entity_fdr_parameters_removed(self, name):
        assert name not in inspect.signature(GDSNavigator.π5_attract_anomaly).parameters

    def test_pi5_default_order_is_delta_norm_desc_without_calibrated_score(self, nav_pi5):
        results, total, _, _ = nav_pi5.π5_attract_anomaly("test_pattern", top_n=20)
        norms = [p.delta_norm for p in results]
        assert norms == sorted(norms, reverse=True)
        assert total == 20


# ======================================================================
# π6 attract_boundary — FDR parameters removed
# ======================================================================


class TestPi6Select:
    def test_pi6_select_diverse(self, nav_pi6):
        results = nav_pi6.π6_attract_boundary(
            "test_alias", "test_pattern", top_n=5, select="diverse"
        )
        for poly, _ in results:
            assert poly.representativeness >= 0  # type: ignore[attr-defined]

    def test_pi6_select_invalid_raises(self, nav_pi6):
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi6.π6_attract_boundary("test_alias", "test_pattern", top_n=5, select="quantum")

    @pytest.mark.parametrize("name", ("fdr_alpha", "fdr_method", "p_value_method"))
    def test_pi6_fdr_parameters_removed(self, name):
        assert name not in inspect.signature(GDSNavigator.π6_attract_boundary).parameters


# ======================================================================
# π7 / π9 — selection only, no FDR
# ======================================================================


class TestPi7Select:
    def test_pi7_select_diverse(self, nav_pi7):
        results = nav_pi7.π7_attract_hub("test_pattern", top_n=5, select="diverse")
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_pi7_select_invalid_raises(self, nav_pi7):
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi7.π7_attract_hub("test_pattern", top_n=5, select="quantum")

    @pytest.mark.parametrize("method_name", ["π7_attract_hub", "π7_attract_hub_and_stats"])
    def test_pi7_has_no_fdr_parameters(self, method_name):
        params = inspect.signature(getattr(GDSNavigator, method_name)).parameters
        assert not {"fdr_alpha", "fdr_method", "p_value_method"} & set(params)


class TestPi9Select:
    def test_pi9_select_diverse(self, nav_pi9):
        for row in nav_pi9.π9_attract_drift("test_pattern", top_n=5, select="diverse"):
            assert row["representativeness"] >= 0

    def test_pi9_select_invalid_raises(self, nav_pi9):
        with pytest.raises(ValueError, match="unknown select mode"):
            nav_pi9.π9_attract_drift("test_pattern", top_n=5, select="quantum")

    def test_pi9_has_no_fdr_parameters(self):
        params = inspect.signature(GDSNavigator.π9_attract_drift).parameters
        assert not {"fdr_alpha", "fdr_method", "p_value_method"} & set(params)


# ======================================================================
# Why entity-level FDR is gone: in-sample conformal p-values have no power
# ======================================================================


class TestEntityFdrRemoved:
    def test_in_sample_conformal_p_never_rejects(self):
        from hypertopos.engine.robust import conformal_p

        rng = np.random.default_rng(0)
        s = rng.normal(size=5_000)
        s[:50] += 6.0
        p = conformal_p(np.sort(s), s)
        for method in ("bh", "storey"):
            rejected, _ = benjamini_hochberg(p, 0.05, method=method)
            assert not rejected.any()

    @pytest.mark.parametrize(
        "name",
        (
            "parametric_p_values_chi2",
            "per_dim_p_values_chi2_univariate",
            "per_dim_p_values_rank_gauss",
            "fdr_per_dimension",
            "empirical_p_values_from_rank",
        ),
    )
    def test_entity_fdr_helpers_removed(self, name):
        assert not hasattr(fdr_module, name)
        assert name not in fdr_module.__all__


# ======================================================================
# Patterns on the calibrated score: selection = is_anomaly, order = anomaly_score
# ======================================================================


def _calibrated_table() -> pa.Table:
    # delta_norm order deliberately differs from anomaly_score order.
    keys = [f"C-{i:02d}" for i in range(6)]
    delta_norm = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    anomaly_score = [0.90, 0.99, 0.95, 0.40, 0.97, 0.30]
    is_anomaly = [True, True, True, False, True, False]
    return pa.table(
        {
            "primary_key": keys,
            "scale": [1] * 6,
            "delta": [[d, 0.0] for d in delta_norm],
            "delta_norm": delta_norm,
            "is_anomaly": is_anomaly,
            "score_rank_pct": [50.0] * 6,
            "conformal_p": [0.04, 0.005, 0.02, 0.5, 0.01, 0.7],
            "anomaly_score": anomaly_score,
            "score_delta": [[1.0, 0.5]] * 6,
            "last_refresh_at": [_DT] * 6,
            "updated_at": [_DT] * 6,
        }
    )


class _MockStorageCalibrated(_MockStoragePi5):
    def __init__(self):
        self._table = _calibrated_table()
        self._n = 6

    def read_sphere(self):
        import dataclasses

        sphere = Sphere("s", "s", ".")
        sphere.patterns["test_pattern"] = dataclasses.replace(
            _PATTERN, population_size=6, score_dims=("a", "b"), score_axes=("mahal", "current")
        )
        return sphere

    def read_reference(self, pattern_id, epoch=None):
        from hypertopos.engine.robust import fit_reference

        rng = np.random.default_rng(0)
        ref, _ = fit_reference(rng.normal(size=(50, 2)), ("a", "b"), axes=("mahal",))
        return ref


@pytest.fixture()
def nav_calibrated(engine):
    return GDSNavigator(engine, _MockStorageCalibrated(), _MANIFEST, _CONTRACT)


class TestCalibratedPattern:
    def test_selects_is_anomaly_and_orders_by_anomaly_score(self, nav_calibrated):
        results, total, _, _ = nav_calibrated.π5_attract_anomaly("test_pattern", top_n=10)
        assert [p.primary_key for p in results] == ["C-01", "C-04", "C-02", "C-00"]
        assert total == 4
        # delta_norm keeps its stored meaning; the score travels separately.
        assert [p.delta_norm for p in results] == [8.0, 5.0, 7.0, 9.0]
        assert [p.anomaly_score for p in results] == [0.99, 0.97, 0.95, 0.90]
        assert results[0].conformal_p == pytest.approx(0.005)

    def test_page_is_ordered_by_p_ascending(self, nav_calibrated):
        results, _, _, _ = nav_calibrated.π5_attract_anomaly("test_pattern", top_n=10)
        ps = [p.conformal_p for p in results]
        assert ps == sorted(ps)

    def test_offset_pages_after_ordering(self, nav_calibrated):
        results, _, _, _ = nav_calibrated.π5_attract_anomaly("test_pattern", top_n=2, offset=1)
        assert [p.primary_key for p in results] == ["C-04", "C-02"]

    @pytest.mark.parametrize(
        "kwargs",
        (
            {"radius": 1.2},
            {"metric": "Linf"},
            {"dimension_weights": {"line_a": 0.5}},
            {"rank_by": "delta_norm"},
            {"rank_by": "signed_confidence"},
            {"boundary_aware": True, "sample_size": 3},
            {"min_confidence": 0.5},
        ),
    )
    def test_shape_space_options_raise(self, nav_calibrated, kwargs):
        with pytest.raises(ValueError, match="calibrated anomaly score"):
            nav_calibrated.π5_attract_anomaly("test_pattern", top_n=5, **kwargs)
