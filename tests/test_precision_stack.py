# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for precision stack: per-group theta, conformal p-values, dimension weights."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypertopos.builder._stats import (
    compute_conformal_p,
    compute_stats_grouped,
)
from hypertopos.builder.builder import GDSBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(tmp_path, n_entities=100, seed=42):
    """Build a minimal sphere with 2 entity types for group testing."""
    rng = np.random.default_rng(seed)

    # Create accounts with 2 entity types having different distributions
    accounts = []
    txs = []
    for i in range(n_entities):
        et = "Corporation" if i < n_entities // 2 else "Individual"
        accounts.append(
            {
                "account_id": f"A{i:04d}",
                "entity_type": et,
            }
        )
        # Corporations have higher tx counts (10-50), Individuals lower (1-10)
        n_tx = rng.integers(10, 50) if et == "Corporation" else rng.integers(1, 10)
        for _ in range(n_tx):
            txs.append(
                {
                    "tx_id": f"T{len(txs):06d}",
                    "from_account": f"A{i:04d}",
                    "amount": float(rng.uniform(10, 10000)),
                }
            )

    b = GDSBuilder("test_sphere", str(tmp_path / "gds"))
    b.add_line("accounts", accounts, key_col="account_id", source_id="test", entity_type="account")
    b.add_line("transactions", txs, key_col="tx_id", source_id="test", role="event")
    b.add_derived_dimension(
        "accounts",
        "transactions",
        "from_account",
        "count",
        None,
        "tx_count",
    )
    b.add_derived_dimension(
        "accounts",
        "transactions",
        "from_account",
        "sum",
        "amount",
        "total_amount",
    )
    b.add_derived_dimension(
        "accounts",
        "transactions",
        "from_account",
        "max",
        "amount",
        "max_amount",
    )
    return b


# ---------------------------------------------------------------------------
# Phase 1A: Per-group theta
# ---------------------------------------------------------------------------


class TestGroupedStats:
    def test_separate_thresholds_per_group(self):
        """Two groups with very different distributions get different theta."""
        rng = np.random.default_rng(42)
        n = 200
        D = 3
        # Group A: tight cluster near 0.5
        group_a = rng.normal(0.5, 0.05, (n, D)).astype(np.float32)
        # Group B: wide spread
        group_b = rng.normal(0.5, 0.3, (n, D)).astype(np.float32)
        shapes = np.vstack([group_a, group_b])
        groups = np.array(["A"] * n + ["B"] * n)

        results, deltas, norms = compute_stats_grouped(shapes, groups, 95.0)
        assert "A" in results
        assert "B" in results

        theta_a_norm = float(np.linalg.norm(results["A"][2]))
        theta_b_norm = float(np.linalg.norm(results["B"][2]))
        # Group A (tight) should have smaller theta than Group B (wide)
        assert theta_a_norm < theta_b_norm, (
            f"Tight group A theta ({theta_a_norm:.3f}) should be < "
            f"wide group B theta ({theta_b_norm:.3f})"
        )

    def test_backward_compat_no_group(self, tmp_path):
        """group_by_property=None produces identical results to ungrouped build."""
        b1 = _make_builder(tmp_path / "a")
        b1.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            tracked_properties=["entity_type"],
        )
        b1.build()

        b2 = _make_builder(tmp_path / "b")
        b2.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            tracked_properties=["entity_type"],
            group_by_property=None,
        )
        b2.build()

        # Both sphere.json should have identical mu/sigma/theta
        s1 = json.loads((tmp_path / "a" / "gds" / "_gds_meta" / "sphere.json").read_text())
        s2 = json.loads((tmp_path / "b" / "gds" / "_gds_meta" / "sphere.json").read_text())
        np.testing.assert_array_almost_equal(
            s1["patterns"]["acct_pattern"]["mu"],
            s2["patterns"]["acct_pattern"]["mu"],
        )
        # No group_stats in ungrouped build
        assert "group_stats" not in s2["patterns"]["acct_pattern"]

    def test_grouped_build_has_group_stats_in_sphere_json(self, tmp_path):
        """Grouped build writes group_stats to sphere.json."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            tracked_properties=["entity_type"],
            group_by_property="entity_type",
        )
        b.build()

        sphere = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
        pat = sphere["patterns"]["acct_pattern"]
        assert "group_stats" in pat
        assert "group_by_property" in pat
        assert pat["group_by_property"] == "entity_type"
        # Should have 2 groups
        assert len(pat["group_stats"]) == 2
        for _gid, gs in pat["group_stats"].items():
            assert "mu" in gs
            assert "sigma_diag" in gs
            assert "theta" in gs
            assert "population_size" in gs

    def test_grouped_stats_serialization_roundtrip(self, tmp_path):
        """group_stats survives sphere.json write → read round-trip."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            tracked_properties=["entity_type"],
            group_by_property="entity_type",
        )
        b.build()

        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(str(tmp_path / "gds"))
        sphere = reader.read_sphere()
        pattern = sphere.patterns["acct_pattern"]
        assert pattern.group_stats is not None
        assert pattern.group_by_property == "entity_type"
        assert len(pattern.group_stats) == 2
        for _gid, gs in pattern.group_stats.items():
            assert gs.mu.shape[0] > 0
            assert gs.sigma_diag.shape[0] > 0
            assert gs.theta.shape[0] > 0
            assert gs.population_size > 0

    def test_invalid_group_by_property_raises(self, tmp_path):
        """group_by_property pointing to non-existent column raises ValueError."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            group_by_property="nonexistent_column",
        )
        with pytest.raises(ValueError, match="nonexistent_column"):
            b.build()

    def test_grouped_deltas_use_group_mean(self):
        """Entities' deltas are computed against their group mean, not global."""
        n = 100
        D = 2
        # Group A centered at [0.2, 0.2], Group B at [0.8, 0.8]
        shapes_a = np.full((n, D), 0.2, dtype=np.float32)
        shapes_b = np.full((n, D), 0.8, dtype=np.float32)
        # Add one outlier in each group
        shapes_a[0] = [0.9, 0.9]  # outlier in A
        shapes_b[0] = [0.1, 0.1]  # outlier in B
        shapes = np.vstack([shapes_a, shapes_b])
        groups = np.array(["A"] * n + ["B"] * n)

        results, deltas, norms = compute_stats_grouped(shapes, groups, 95.0)

        # Outlier in group A (index 0) should have high norm
        assert norms[0] > norms[1], "Outlier should have higher norm than regular entity"
        # Outlier in group B (index n) should have high norm
        assert norms[n] > norms[n + 1], "Outlier should have higher norm than regular entity"

        # Regular entities should have near-zero delta norm (all identical within group)
        assert norms[1] < 1.0, f"Regular entity should have low norm, got {norms[1]}"


# ---------------------------------------------------------------------------
# Phase 1B: Conformal p-values
# ---------------------------------------------------------------------------


class TestConformalP:
    def test_range(self):
        """Conformal p-values should be in (0, 1]."""
        norms = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        p = compute_conformal_p(norms)
        assert p.min() > 0, f"min p should be > 0, got {p.min()}"
        assert p.max() <= 1.0, f"max p should be <= 1, got {p.max()}"

    def test_monotonic(self):
        """Higher delta_norm → lower (or equal) conformal p-value."""
        norms = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        p = compute_conformal_p(norms)
        # Sort norms ascending, p should be descending
        sorted_indices = np.argsort(norms)
        sorted_p = p[sorted_indices]
        for i in range(len(sorted_p) - 1):
            assert sorted_p[i] >= sorted_p[i + 1], (
                f"p[{i}]={sorted_p[i]} should be >= p[{i + 1}]={sorted_p[i + 1]}"
            )

    def test_extreme_values(self):
        """Entity with highest norm gets lowest p, entity with lowest norm gets highest p."""
        norms = np.array([10.0, 1.0, 5.0, 3.0, 7.0], dtype=np.float32)
        p = compute_conformal_p(norms)
        assert p[np.argmax(norms)] == p.min()
        assert p[np.argmin(norms)] == p.max()

    def test_conformal_p_in_geometry(self, tmp_path):
        """Built sphere geometry includes conformal_p column."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
        )
        b.build()

        import lance

        geo = lance.dataset(
            str(tmp_path / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
        ).to_table(columns=["conformal_p"])
        assert "conformal_p" in geo.schema.names
        p_vals = geo["conformal_p"].to_numpy()
        assert p_vals.min() > 0
        assert p_vals.max() <= 1.0

    def test_empty_norms(self):
        """Empty input returns empty output."""
        p = compute_conformal_p(np.array([], dtype=np.float32))
        assert len(p) == 0


# ---------------------------------------------------------------------------
# Phase 2: Weighted dimensions
# ---------------------------------------------------------------------------


class TestDimensionWeights:
    def test_kurtosis_weights_amplify_heavy_tails(self):
        """Dim with heavy tail gets higher weight than uniform dim."""
        from hypertopos.builder._stats import compute_dimension_weights

        rng = np.random.default_rng(42)
        n = 1000
        # Dim 0: uniform (low kurtosis)
        d0 = rng.uniform(0, 1, n)
        # Dim 1: heavy-tailed (high kurtosis) — exponential
        d1 = rng.exponential(0.1, n)
        d1 = np.clip(d1, 0, 1)
        shapes = np.column_stack([d0, d1]).astype(np.float32)

        w = compute_dimension_weights(shapes, method="kurtosis")
        assert w.shape == (2,)
        assert w[0] >= 1.0
        assert w[1] >= 1.0
        # Exponential has higher kurtosis than uniform
        assert w[1] > w[0], f"Heavy-tail dim weight ({w[1]}) should exceed uniform ({w[0]})"

    def test_uniform_weights_all_ones(self):
        """Method='uniform' returns all 1.0."""
        from hypertopos.builder._stats import compute_dimension_weights

        shapes = np.random.randn(100, 5).astype(np.float32)
        w = compute_dimension_weights(shapes, method="uniform")
        np.testing.assert_array_equal(w, np.ones(5, dtype=np.float32))

    def test_weighted_build_changes_norms(self, tmp_path):
        """dimension_weights='auto' produces different delta_norms than unweighted."""
        b1 = _make_builder(tmp_path / "a")
        b1.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
        )
        b1.build()

        b2 = _make_builder(tmp_path / "b")
        b2.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            dimension_weights="auto",
        )
        b2.build()

        import lance

        norms1 = (
            lance.dataset(
                str(tmp_path / "a" / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            )
            .to_table(columns=["delta_norm"])["delta_norm"]
            .to_numpy()
        )
        norms2 = (
            lance.dataset(
                str(tmp_path / "b" / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            )
            .to_table(columns=["delta_norm"])["delta_norm"]
            .to_numpy()
        )

        # Norms should differ (weights != uniform)
        assert not np.allclose(norms1, norms2), "Weighted norms should differ from unweighted"

    def test_weighted_backward_compat(self, tmp_path):
        """dimension_weights=None produces identical results."""
        b1 = _make_builder(tmp_path / "a")
        b1.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            dimension_weights=None,
        )
        b1.build()

        b2 = _make_builder(tmp_path / "b")
        b2.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
        )
        b2.build()

        import lance

        norms1 = (
            lance.dataset(
                str(tmp_path / "a" / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            )
            .to_table(columns=["delta_norm"])["delta_norm"]
            .to_numpy()
        )
        norms2 = (
            lance.dataset(
                str(tmp_path / "b" / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            )
            .to_table(columns=["delta_norm"])["delta_norm"]
            .to_numpy()
        )

        np.testing.assert_array_almost_equal(norms1, norms2)

    def test_weights_serialized_in_sphere_json(self, tmp_path):
        """dimension_weights='auto' persists weights in sphere.json."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            dimension_weights="auto",
        )
        b.build()

        sphere = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
        pat = sphere["patterns"]["acct_pattern"]
        assert "dimension_weights" in pat
        assert len(pat["dimension_weights"]) > 0
        assert all(w >= 1.0 for w in pat["dimension_weights"])


# ---------------------------------------------------------------------------
# Phase 3: GMM per-cluster theta
# ---------------------------------------------------------------------------


class TestGMM:
    def test_gmm_creates_components(self):
        """fit_kmeans_components returns k components."""
        from hypertopos.builder._stats import fit_kmeans_components

        rng = np.random.default_rng(42)
        shapes = rng.random((200, 3)).astype(np.float32)
        components, assignments = fit_kmeans_components(shapes, n_components=3)
        assert len(components) == 3
        assert assignments.shape == (200,)
        assert set(assignments.tolist()) <= {0, 1, 2}
        for mu_c, sig_c, th_c, pop_c in components:
            assert mu_c.shape == (3,)
            assert sig_c.shape == (3,)
            assert th_c.shape == (3,)
            assert pop_c > 0

    def test_gmm_build_serialization(self, tmp_path):
        """GMM components serialized in sphere.json."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            gmm_n_components=3,
        )
        b.build()

        sphere = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
        pat = sphere["patterns"]["acct_pattern"]
        assert "gmm_components" in pat
        assert len(pat["gmm_components"]) == 3
        for comp in pat["gmm_components"]:
            assert "mu" in comp
            assert "sigma_diag" in comp
            assert "theta" in comp
            assert "population_size" in comp

    def test_gmm_deserialization_roundtrip(self, tmp_path):
        """GMM components survive write → read round-trip."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            gmm_n_components=3,
        )
        b.build()

        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(str(tmp_path / "gds"))
        sphere = reader.read_sphere()
        pattern = sphere.patterns["acct_pattern"]
        assert pattern.gmm_components is not None
        assert len(pattern.gmm_components) == 3
        for comp in pattern.gmm_components:
            assert comp.mu.shape[0] > 0
            assert comp.population_size > 0

    def test_gmm_backward_compat(self, tmp_path):
        """gmm_n_components=None → no gmm_components in sphere.json."""
        b = _make_builder(tmp_path)
        b.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
        )
        b.build()

        sphere = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
        assert "gmm_components" not in sphere["patterns"]["acct_pattern"]

    def test_gmm_changes_delta_norms(self, tmp_path):
        """GMM produces different delta_norms than global stats."""
        b1 = _make_builder(tmp_path / "a")
        b1.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
        )
        b1.build()

        b2 = _make_builder(tmp_path / "b")
        b2.add_pattern(
            "acct_pattern",
            "anchor",
            "accounts",
            relations=[],
            anomaly_percentile=95.0,
            gmm_n_components=3,
        )
        b2.build()

        import lance

        norms1 = (
            lance.dataset(
                str(tmp_path / "a" / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            )
            .to_table(columns=["delta_norm"])["delta_norm"]
            .to_numpy()
        )
        norms2 = (
            lance.dataset(
                str(tmp_path / "b" / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            )
            .to_table(columns=["delta_norm"])["delta_norm"]
            .to_numpy()
        )

        assert not np.allclose(norms1, norms2), "GMM norms should differ from global"


# ---------------------------------------------------------------------------
# Cross-pattern entity profile
# ---------------------------------------------------------------------------
# Step 3: Inter-event-time features
# ---------------------------------------------------------------------------


class TestIET:
    def test_iet_mean(self, tmp_path):
        """IET mean computes average time between events per entity."""
        b = GDSBuilder("iet_test", str(tmp_path / "gds"))
        b.add_line(
            "accounts",
            [
                {"account_id": "A1"},
                {"account_id": "A2"},
            ],
            key_col="account_id",
            source_id="test",
        )
        b.add_line(
            "txs",
            [
                {"tx_id": "T1", "from": "A1", "ts": "2025-01-01T00:00:00"},
                {"tx_id": "T2", "from": "A1", "ts": "2025-01-01T01:00:00"},
                {"tx_id": "T3", "from": "A1", "ts": "2025-01-01T03:00:00"},
                {"tx_id": "T4", "from": "A2", "ts": "2025-01-01T00:00:00"},
            ],
            key_col="tx_id",
            source_id="test",
            role="event",
        )
        b.add_derived_dimension(
            "accounts",
            "txs",
            "from",
            "iet_mean",
            None,
            "avg_gap",
            time_col="ts",
        )
        b.add_pattern("p", "anchor", "accounts", relations=[])
        b.build()

        import lance

        geo = lance.dataset(
            str(tmp_path / "gds" / "geometry" / "p" / "v=1" / "data.lance")
        ).to_table(columns=["primary_key", "delta_norm"])
        assert geo.num_rows == 2

    def test_iet_requires_time_col(self):
        """IET metric without time_col raises ValueError."""
        b = GDSBuilder("iet_err", "tmp")
        b.add_line("a", [{"id": "1"}], key_col="id", source_id="test")
        b.add_line("e", [{"eid": "1", "fk": "1"}], key_col="eid", source_id="test", role="event")
        with pytest.raises(ValueError, match="time_col"):
            b.add_derived_dimension("a", "e", "fk", "iet_mean", None, "gap")
            b.add_pattern("p", "anchor", "a", relations=[])
            b.build()


# ---------------------------------------------------------------------------
# Step 1: Mahalanobis distance
# ---------------------------------------------------------------------------


class TestMahalanobis:
    def test_correlated_dims_differ(self):
        """Correlated dims produce different norms under Mahalanobis vs diagonal."""
        from hypertopos.builder._stats import compute_stats

        rng = np.random.default_rng(42)
        n = 500
        # Correlated: dim0 and dim1 move together (ρ ≈ 0.9)
        base = rng.normal(0.5, 0.1, n)
        d0 = base + rng.normal(0, 0.01, n)
        d1 = base + rng.normal(0, 0.01, n)
        d2 = rng.uniform(0, 1, n)  # uncorrelated
        shapes = np.column_stack([d0, d1, d2]).astype(np.float32)

        _, _, _, _, norms_diag, _ = compute_stats(shapes, 95.0, use_mahalanobis=False)
        _, _, _, _, norms_mah, cov_inv = compute_stats(shapes, 95.0, use_mahalanobis=True)

        assert cov_inv is not None
        assert cov_inv.shape == (3, 3)
        assert not np.allclose(norms_diag, norms_mah), (
            "Mahalanobis should differ for correlated data"
        )

    def test_backward_compat(self, tmp_path):
        """use_mahalanobis=False produces identical results."""
        b = _make_builder(tmp_path)
        b.add_pattern("acct_pattern", "anchor", "accounts", relations=[], use_mahalanobis=False)
        b.build()

        import lance

        norms = (
            lance.dataset(
                str(tmp_path / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            )
            .to_table(columns=["delta_norm"])["delta_norm"]
            .to_numpy()
        )
        assert len(norms) > 0

    def test_mahalanobis_build_and_serialize(self, tmp_path):
        """Mahalanobis cholesky_inv persists in sphere.json."""
        b = _make_builder(tmp_path)
        b.add_pattern("acct_pattern", "anchor", "accounts", relations=[], use_mahalanobis=True)
        b.build()

        sphere = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
        pat = sphere["patterns"]["acct_pattern"]
        assert "cholesky_inv" in pat
        cov = pat["cholesky_inv"]
        assert len(cov) > 0
        assert len(cov[0]) > 0  # 2D matrix

    def test_per_dim_anomaly_count_basic(self):
        """Entity with 2/5 dims above p99 gets count=2."""
        from hypertopos.builder._stats import compute_per_dim_anomaly_count

        rng = np.random.default_rng(42)
        deltas = rng.normal(0, 1, (200, 5)).astype(np.float32)
        # Make entity 0 extreme in 2 dims
        deltas[0, 0] = 10.0
        deltas[0, 1] = 10.0
        counts = compute_per_dim_anomaly_count(deltas, percentile=99.0)
        assert counts[0] >= 2, f"Expected >=2 anomalous dims, got {counts[0]}"
        # Regular entities should have 0 or 1
        assert counts[1:].mean() < 1.0

    def test_n_anomalous_dims_in_geometry(self, tmp_path):
        """Built sphere geometry includes n_anomalous_dims column."""
        b = _make_builder(tmp_path)
        b.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
        b.build()

        import lance

        geo = lance.dataset(
            str(tmp_path / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
        ).to_table(columns=["n_anomalous_dims"])
        assert "n_anomalous_dims" in geo.schema.names
        vals = geo["n_anomalous_dims"].to_numpy()
        assert vals.min() >= 0

    def test_mahalanobis_roundtrip(self, tmp_path):
        """cholesky_inv survives write → read round-trip."""
        b = _make_builder(tmp_path)
        b.add_pattern("acct_pattern", "anchor", "accounts", relations=[], use_mahalanobis=True)
        b.build()

        from hypertopos.storage.reader import GDSReader

        reader = GDSReader(str(tmp_path / "gds"))
        sphere = reader.read_sphere()
        pattern = sphere.patterns["acct_pattern"]
        assert pattern.cholesky_inv is not None
        assert pattern.cholesky_inv.ndim == 2


# ---------------------------------------------------------------------------
# Cross-pattern entity profile
# ---------------------------------------------------------------------------


def _make_multi_pattern_builder(tmp_path, n=50, seed=42):
    """Build sphere with account + pair + chain patterns."""
    rng = np.random.default_rng(seed)
    from hypertopos.engine.chains import Chain

    accounts = [{"account_id": f"A{i:04d}", "entity_type": "Individual"} for i in range(n)]

    # Transactions between accounts
    txs = []
    for i in range(n * 5):
        src = f"A{rng.integers(0, n):04d}"
        dst = f"A{rng.integers(0, n):04d}"
        if src == dst:
            continue
        txs.append(
            {
                "tx_id": f"T{i:06d}",
                "from_account": src,
                "to_account": dst,
                "amount": float(rng.uniform(100, 10000)),
                "timestamp": f"2025-01-{(i % 28) + 1:02d}T12:00:00",
            }
        )

    b = GDSBuilder("multi_test", str(tmp_path / "gds"))
    b.add_line("accounts", accounts, key_col="account_id", source_id="test", entity_type="account")
    b.add_line("transactions", txs, key_col="tx_id", source_id="test", role="event")

    # Account pattern (direct)
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "tx_out")
    b.add_derived_dimension("accounts", "transactions", "to_account", "count", None, "tx_in")
    b.add_pattern("account_pattern", "anchor", "accounts", relations=[])

    # Pair pattern (composite)
    b.add_composite_line("account_pairs", "transactions", ["from_account", "to_account"])
    b.add_derived_dimension(
        "account_pairs", "transactions", ["from_account", "to_account"], "count", None, "pair_count"
    )
    b.add_pattern("pair_pattern", "anchor", "account_pairs", relations=[])

    # Chain pattern
    chains = []
    for ci in range(20):
        keys = [f"A{rng.integers(0, n):04d}" for _ in range(3)]
        c = Chain(
            chain_id=f"ch_{ci:03d}",
            keys=keys,
            event_keys=[f"T{ci:03d}_{j}" for j in range(2)],
            hop_count=2,
            is_cyclic=(keys[0] == keys[-1]),
            time_span_hours=10.0,
            categories=["USD", "USD"],
            amounts=[100.0, 90.0],
            amount_decay=0.9,
        )
        chains.append(c.to_dict())
    b.add_chain_line("tx_chains", chains, features=["hop_count", "is_cyclic"])

    return b


class TestCrossPatternProfile:
    def test_direct_pattern_only(self, tmp_path):
        """Single-pattern sphere returns one direct signal."""
        b = _make_builder(tmp_path)
        b.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
        b.build()

        from hypertopos.sphere import HyperSphere

        hs = HyperSphere.open(str(tmp_path / "gds"))
        with hs.session("test") as sess:
            nav = sess.navigator()
            profile = nav.cross_pattern_profile("A0000", line_id="accounts")
            assert profile["primary_key"] == "A0000"
            assert profile["total_patterns"] == 1
            assert "acct_pattern" in profile["signals"]
            sig = profile["signals"]["acct_pattern"]
            assert sig["key_type"] == "direct"
            assert isinstance(sig["is_anomaly"], bool)
            assert isinstance(sig["delta_norm"], float)

    def test_multi_pattern_discovers_all(self, tmp_path):
        """Multi-pattern sphere discovers direct + composite + chain patterns."""
        b = _make_multi_pattern_builder(tmp_path)
        b.build()

        from hypertopos.sphere import HyperSphere

        hs = HyperSphere.open(str(tmp_path / "gds"))
        with hs.session("test") as sess:
            nav = sess.navigator()
            profile = nav.cross_pattern_profile("A0000", line_id="accounts")
            assert profile["total_patterns"] >= 1
            assert "account_pattern" in profile["signals"]
            acct_sig = profile["signals"]["account_pattern"]
            assert acct_sig["key_type"] == "direct"

    def test_source_count_correct(self, tmp_path):
        """source_count correctly counts patterns with at least one anomaly."""
        b = _make_builder(tmp_path)
        b.add_pattern("acct_pattern", "anchor", "accounts", relations=[], anomaly_percentile=50.0)
        b.build()

        from hypertopos.sphere import HyperSphere

        hs = HyperSphere.open(str(tmp_path / "gds"))
        with hs.session("test") as sess:
            nav = sess.navigator()

            import lance

            geo = lance.dataset(
                str(tmp_path / "gds" / "geometry" / "acct_pattern" / "v=1" / "data.lance")
            ).to_table(columns=["primary_key", "is_anomaly"])
            anom_keys = [
                geo["primary_key"][i].as_py()
                for i in range(geo.num_rows)
                if geo["is_anomaly"][i].as_py()
            ]
            if anom_keys:
                profile = nav.cross_pattern_profile(anom_keys[0], line_id="accounts")
                assert profile["source_count"] >= 1

    def test_entity_not_found(self, tmp_path):
        """Unknown entity raises error."""
        b = _make_builder(tmp_path)
        b.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
        b.build()

        from hypertopos.navigation.navigator import GDSEntityNotFoundError
        from hypertopos.sphere import HyperSphere

        hs = HyperSphere.open(str(tmp_path / "gds"))
        with hs.session("test") as sess:
            nav = sess.navigator()
            with pytest.raises(GDSEntityNotFoundError):
                nav.cross_pattern_profile("NONEXISTENT_KEY")
