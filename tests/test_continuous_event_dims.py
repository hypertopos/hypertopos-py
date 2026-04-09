# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for continuous event dimensions."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from hypertopos.builder.builder import GDSBuilder, RelationSpec


@pytest.fixture(scope="module")
def event_sphere(tmp_path_factory):
    """Build a small sphere with mixed binary + continuous event dims."""
    tmp = tmp_path_factory.mktemp("event_dims")
    out_path = str(tmp / "gds_test")

    # Create anchor line (customers)
    customers = pa.table(
        {
            "primary_key": pa.array([f"C{i}" for i in range(100)], type=pa.string()),
        }
    )

    # Create event line with amounts
    # 199 normal events (amount 10-100), 1 outlier (amount 5000)
    rng = np.random.default_rng(42)
    n_events = 200
    amounts = rng.uniform(10.0, 100.0, size=n_events).tolist()
    amounts[0] = 5000.0  # outlier

    cust_fks = [f"C{rng.integers(0, 100)}" for _ in range(n_events)]
    cust_fks[0] = "C0"  # ensure outlier has valid FK

    events = pa.table(
        {
            "primary_key": pa.array([f"E{i}" for i in range(n_events)], type=pa.string()),
            "customer_id": pa.array(cust_fks, type=pa.string()),
            "amount": pa.array(amounts, type=pa.float64()),
        }
    )

    builder = GDSBuilder("test_event_dims", out_path)
    builder.add_line("customers", customers, key_col="primary_key", source_id="test", role="anchor")
    builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")

    builder.add_pattern(
        "event_pattern",
        "event",
        "events",
        relations=[
            RelationSpec(
                line_id="customers",
                fk_col="customer_id",
                direction="in",
                required=True,
            ),
        ],
    )
    builder.add_event_dimension("event_pattern", "amount", edge_max="auto")
    builder.build()
    return out_path, amounts


@pytest.fixture(scope="module")
def binary_only_sphere(tmp_path_factory):
    """Build a sphere with binary-only event pattern (backward compat)."""
    tmp = tmp_path_factory.mktemp("binary_event")
    out_path = str(tmp / "gds_binary")

    customers = pa.table(
        {
            "primary_key": pa.array([f"C{i}" for i in range(50)], type=pa.string()),
        }
    )
    suppliers = pa.table(
        {
            "primary_key": pa.array([f"S{i}" for i in range(20)], type=pa.string()),
        }
    )

    rng = np.random.default_rng(99)
    n = 100
    events = pa.table(
        {
            "primary_key": pa.array([f"E{i}" for i in range(n)], type=pa.string()),
            "cust_fk": pa.array([f"C{rng.integers(0, 50)}" for _ in range(n)], type=pa.string()),
            "supp_fk": pa.array([f"S{rng.integers(0, 20)}" for _ in range(n)], type=pa.string()),
        }
    )

    builder = GDSBuilder("test_binary", out_path)
    builder.add_line("customers", customers, key_col="primary_key", source_id="test", role="anchor")
    builder.add_line("suppliers", suppliers, key_col="primary_key", source_id="test", role="anchor")
    builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")

    builder.add_pattern(
        "binary_event_pattern",
        "event",
        "events",
        relations=[
            RelationSpec(
                line_id="customers",
                fk_col="cust_fk",
                direction="in",
                required=True,
            ),
            RelationSpec(
                line_id="suppliers",
                fk_col="supp_fk",
                direction="in",
                required=True,
            ),
        ],
    )
    builder.build()
    return out_path


class TestMixedBinaryContinuousDims:
    """Test 1: Mixed binary + continuous dims."""

    def test_delta_has_two_dimensions(self, event_sphere):
        out_path, _ = event_sphere
        geo_path = str(Path(out_path) / "geometry" / "event_pattern" / "v=1" / "data.lance")
        geo = lance.dataset(geo_path).to_table()
        deltas = geo["delta"].to_pylist()

        # Every delta vector must have exactly 2 dimensions
        for i, d in enumerate(deltas):
            assert len(d) == 2, f"Entity {i}: expected 2-dim delta, got {len(d)}"

    def test_binary_dim_values_are_zero_or_one(self, event_sphere):
        out_path, _ = event_sphere
        geo_path = str(Path(out_path) / "geometry" / "event_pattern" / "v=1" / "data.lance")
        geo = lance.dataset(geo_path).to_table()

        # Reconstruct shape vectors from delta: shape = delta * sigma + mu
        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["event_pattern"]
        mu = np.array(pat["mu"], dtype=np.float32)
        sigma = np.array(pat["sigma_diag"], dtype=np.float32)

        deltas = np.array(geo["delta"].to_pylist(), dtype=np.float32)
        shapes = deltas * sigma + mu

        # Binary FK dim (index 0): shape values must be 0.0 or 1.0
        binary_vals = set(np.unique(np.round(shapes[:, 0], 4)))
        assert binary_vals <= {0.0, 1.0}, (
            f"Binary dim (customer FK) should only be 0.0 or 1.0, got unique values: {binary_vals}"
        )

    def test_continuous_dim_in_range_zero_to_three(self, event_sphere):
        out_path, _ = event_sphere
        geo_path = str(Path(out_path) / "geometry" / "event_pattern" / "v=1" / "data.lance")
        geo = lance.dataset(geo_path).to_table()

        # Reconstruct shape vectors from delta
        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["event_pattern"]
        mu = np.array(pat["mu"], dtype=np.float32)
        sigma = np.array(pat["sigma_diag"], dtype=np.float32)

        deltas = np.array(geo["delta"].to_pylist(), dtype=np.float32)
        shapes = deltas * sigma + mu

        # Continuous dim (index 1): shape values in [0.0, 3.0] (clipped)
        continuous_vals = shapes[:, 1]
        assert np.all(continuous_vals >= -0.01), (
            f"Continuous dim values below 0.0: min={continuous_vals.min():.6f}"
        )
        assert np.all(continuous_vals <= 3.01), (
            f"Continuous dim values above 3.0: max={continuous_vals.max():.6f}"
        )

    def test_continuous_dim_has_non_binary_values(self, event_sphere):
        out_path, _ = event_sphere
        geo_path = str(Path(out_path) / "geometry" / "event_pattern" / "v=1" / "data.lance")
        geo = lance.dataset(geo_path).to_table()

        # Reconstruct shape vectors from delta
        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["event_pattern"]
        mu = np.array(pat["mu"], dtype=np.float32)
        sigma = np.array(pat["sigma_diag"], dtype=np.float32)

        deltas = np.array(geo["delta"].to_pylist(), dtype=np.float32)
        shapes = deltas * sigma + mu

        # Continuous dim (index 1) should have non-binary values
        continuous_unique = set(np.unique(np.round(shapes[:, 1], 4)))
        non_binary = continuous_unique - {0.0, 1.0}
        assert len(non_binary) > 0, (
            "Continuous dim should have non-binary values (not just 0/1), "
            f"got unique values: {continuous_unique}"
        )


class TestAnomalyDetectsValueOutlier:
    """Test 2: Anomaly detection picks up value outlier."""

    def test_outlier_is_anomaly(self, event_sphere):
        out_path, _ = event_sphere
        geo_path = str(Path(out_path) / "geometry" / "event_pattern" / "v=1" / "data.lance")
        geo = lance.dataset(geo_path).to_table()

        pks = geo["primary_key"].to_pylist()
        anomaly_flags = geo["is_anomaly"].to_pylist()

        # E0 is the outlier (amount=5000)
        e0_idx = pks.index("E0")
        assert anomaly_flags[e0_idx] is True, (
            f"E0 (amount=5000) should be flagged as anomaly, but is_anomaly={anomaly_flags[e0_idx]}"
        )


class TestEdgeMaxAutoP99:
    """Test 3: edge_max='auto' computes 99th percentile correctly."""

    def test_edge_max_matches_p99(self, event_sphere):
        out_path, amounts = event_sphere

        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["event_pattern"]
        stored_edge_max_list = pat["edge_max"]

        # Index 1 = the continuous amount dimension
        stored_em = stored_edge_max_list[1]

        # Compute expected p99 from the positive amounts
        amounts_arr = np.array(amounts, dtype=np.float32)
        positive = amounts_arr[amounts_arr > 0]
        expected_p99 = float(np.percentile(positive, 99.0))

        assert stored_em == pytest.approx(expected_p99, rel=1e-3), (
            f"edge_max for amount dim should be ~p99={expected_p99:.4f}, got {stored_em}"
        )

    def test_edge_max_stored_in_sphere_json(self, event_sphere):
        out_path, _ = event_sphere

        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["event_pattern"]

        # edge_max list must exist (not None) when event dimensions present
        assert pat["edge_max"] is not None, (
            "sphere.json edge_max must be a list when event dimensions are used"
        )
        # Must have 2 entries (1 relation + 1 event dimension)
        assert len(pat["edge_max"]) == 2

    def test_event_dimensions_persisted_in_sphere_json(self, event_sphere):
        out_path, _ = event_sphere

        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["event_pattern"]

        # event_dimensions should be present with amount column
        edims = pat.get("event_dimensions", [])
        assert len(edims) == 1, f"Expected 1 event_dimension, got {len(edims)}: {edims}"
        assert edims[0]["column"] == "amount", (
            f"Expected event_dimension column='amount', got {edims[0]}"
        )


class TestBinaryOnlyBackwardCompat:
    """Test 4: Binary-only event pattern unchanged (backward compat)."""

    def test_all_deltas_binary(self, binary_only_sphere):
        out_path = binary_only_sphere
        geo_path = str(Path(out_path) / "geometry" / "binary_event_pattern" / "v=1" / "data.lance")
        geo = lance.dataset(geo_path).to_table()

        # Reconstruct shape vectors: shape = delta * sigma + mu
        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["binary_event_pattern"]
        mu = np.array(pat["mu"], dtype=np.float32)
        sigma = np.array(pat["sigma_diag"], dtype=np.float32)

        deltas = np.array(geo["delta"].to_pylist(), dtype=np.float32)
        shapes = deltas * sigma + mu

        for i in range(shapes.shape[0]):
            for dim_idx in range(shapes.shape[1]):
                val = round(float(shapes[i, dim_idx]), 4)
                assert val in (0.0, 1.0), (
                    f"Entity {i}, dim {dim_idx}: expected 0.0 or 1.0, got {val}"
                )

    def test_no_edge_max_in_sphere_json(self, binary_only_sphere):
        out_path = binary_only_sphere
        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["binary_event_pattern"]
        assert pat.get("edge_max") is None, (
            f"Binary-only pattern should have edge_max=None, got {pat.get('edge_max')}"
        )

    def test_no_event_dimensions(self, binary_only_sphere):
        out_path = binary_only_sphere
        sphere_json = json.loads((Path(out_path) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["binary_event_pattern"]
        edims = pat.get("event_dimensions", [])
        assert edims == [], f"Binary-only pattern should have empty event_dimensions, got {edims}"


class TestEventDimValidation:
    """Test: validation errors for invalid event dimensions."""

    def test_event_dim_on_anchor_pattern_raises(self, tmp_path):
        builder = GDSBuilder("test_val", str(tmp_path / "gds_val"))
        builder.add_line(
            "custs",
            pa.table({"primary_key": pa.array(["C1"], type=pa.string())}),
            key_col="primary_key",
            source_id="test",
            role="anchor",
        )
        builder.add_pattern("cust_pat", "anchor", "custs", relations=[])
        with pytest.raises(ValueError, match="only applies to event"):
            builder.add_event_dimension("cust_pat", "amount")

    def test_event_dim_missing_column_raises(self, tmp_path):
        builder = GDSBuilder("test_val2", str(tmp_path / "gds_val2"))
        builder.add_line(
            "custs",
            pa.table({"primary_key": pa.array(["C1"], type=pa.string())}),
            key_col="primary_key",
            source_id="test",
            role="anchor",
        )
        builder.add_line(
            "evts",
            pa.table(
                {
                    "primary_key": pa.array(["E1"], type=pa.string()),
                    "cust_fk": pa.array(["C1"], type=pa.string()),
                }
            ),
            key_col="primary_key",
            source_id="test",
            role="event",
        )
        builder.add_pattern(
            "evt_pat",
            "event",
            "evts",
            relations=[RelationSpec("custs", "cust_fk", "in")],
        )
        builder.add_event_dimension("evt_pat", "nonexistent_col")
        with pytest.raises(ValueError, match="not found"):
            builder.build()

    def test_event_dim_non_numeric_column_raises(self, tmp_path):
        builder = GDSBuilder("test_val3", str(tmp_path / "gds_val3"))
        builder.add_line(
            "custs",
            pa.table({"primary_key": pa.array(["C1"], type=pa.string())}),
            key_col="primary_key",
            source_id="test",
            role="anchor",
        )
        builder.add_line(
            "evts",
            pa.table(
                {
                    "primary_key": pa.array(["E1"], type=pa.string()),
                    "cust_fk": pa.array(["C1"], type=pa.string()),
                    "name": pa.array(["hello"], type=pa.string()),
                }
            ),
            key_col="primary_key",
            source_id="test",
            role="event",
        )
        builder.add_pattern(
            "evt_pat",
            "event",
            "evts",
            relations=[RelationSpec("custs", "cust_fk", "in")],
        )
        builder.add_event_dimension("evt_pat", "name")
        with pytest.raises(ValueError, match="must be numeric"):
            builder.build()


class TestMultipleEventDims:
    """Test: multiple event dimensions + explicit edge_max."""

    def test_two_event_dims(self, tmp_path):
        rng = np.random.default_rng(77)
        n = 100
        events = pa.table(
            {
                "primary_key": pa.array([f"E{i}" for i in range(n)], type=pa.string()),
                "amount": pa.array(rng.uniform(10, 100, size=n).tolist(), type=pa.float64()),
                "quantity": pa.array(rng.uniform(1, 50, size=n).tolist(), type=pa.float64()),
            }
        )

        out = str(tmp_path / "gds_multi")
        builder = GDSBuilder("test_multi", out)
        builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")
        builder.add_pattern("ep", "event", "events", relations=[])
        builder.add_event_dimension("ep", "amount", edge_max="auto")
        builder.add_event_dimension("ep", "quantity", edge_max="auto")
        builder.build()

        sphere_json = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["ep"]
        assert len(pat["mu"]) == 2, f"Expected 2 dims, got {len(pat['mu'])}"
        assert len(pat["event_dimensions"]) == 2
        assert pat["event_dimensions"][0]["column"] == "amount"
        assert pat["event_dimensions"][1]["column"] == "quantity"
        assert pat["edge_max"] is not None
        assert len(pat["edge_max"]) == 2

    def test_explicit_edge_max_float(self, tmp_path):
        rng = np.random.default_rng(88)
        n = 100
        events = pa.table(
            {
                "primary_key": pa.array([f"E{i}" for i in range(n)], type=pa.string()),
                "amount": pa.array(rng.uniform(10, 100, size=n).tolist(), type=pa.float64()),
            }
        )

        out = str(tmp_path / "gds_explicit")
        builder = GDSBuilder("test_explicit", out)
        builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")
        builder.add_pattern("ep", "event", "events", relations=[])
        builder.add_event_dimension("ep", "amount", edge_max=200.0)
        builder.build()

        sphere_json = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
        pat = sphere_json["patterns"]["ep"]
        # edge_max should be exactly 200.0 (not auto-computed)
        assert pat["event_dimensions"][0]["edge_max"] == 200.0
        assert pat["edge_max"][0] == 200.0

    def test_dim_index_resolves_event_dim(self, tmp_path):
        """Verify Pattern.dim_index finds event dims by column name."""
        rng = np.random.default_rng(99)
        n = 50
        custs = pa.table(
            {
                "primary_key": pa.array([f"C{i}" for i in range(20)], type=pa.string()),
            }
        )
        events = pa.table(
            {
                "primary_key": pa.array([f"E{i}" for i in range(n)], type=pa.string()),
                "cust_fk": pa.array(
                    [f"C{rng.integers(0, 20)}" for _ in range(n)], type=pa.string()
                ),
                "price": pa.array(rng.uniform(1, 100, size=n).tolist(), type=pa.float64()),
            }
        )

        out = str(tmp_path / "gds_dimidx")
        builder = GDSBuilder("test_dimidx", out)
        builder.add_line("custs", custs, key_col="primary_key", source_id="test", role="anchor")
        builder.add_line("events", events, key_col="primary_key", source_id="test", role="event")
        builder.add_pattern(
            "ep",
            "event",
            "events",
            relations=[RelationSpec("custs", "cust_fk", "in")],
        )
        builder.add_event_dimension("ep", "price", edge_max="auto")
        builder.build()

        from hypertopos.sphere import HyperSphere

        sphere = HyperSphere.open(out)
        pattern = sphere._sphere.patterns["ep"]

        # Relation dim at index 0
        assert pattern.dim_index("custs") == 0
        # Event dim at index 1
        assert pattern.dim_index("price") == 1
        # dim_labels should reflect both
        assert pattern.dim_labels == ["custs", "price"]


class TestFindAnomaliesIntegration:
    """Test 5: find_anomalies returns anomalous events by value."""

    def test_find_anomalies_returns_value_outlier(self, event_sphere):
        from hypertopos.sphere import HyperSphere

        out_path, _ = event_sphere
        sphere = HyperSphere.open(out_path)
        session = sphere.session("test_anomalies")
        nav = session.navigator()

        anomalies, total_found, _, _ = nav.π5_attract_anomaly("event_pattern", top_n=10)

        anomaly_keys = {p.primary_key for p in anomalies}
        assert "E0" in anomaly_keys, (
            f"E0 (amount=5000 outlier) should appear in top-10 anomalies, got {anomaly_keys}"
        )
        assert total_found > 0, "Expected at least 1 anomaly found"

        session.close()

    def test_outlier_has_high_delta_norm(self, event_sphere):
        from hypertopos.sphere import HyperSphere

        out_path, _ = event_sphere
        sphere = HyperSphere.open(out_path)
        session = sphere.session("test_norm")
        nav = session.navigator()

        anomalies, _, _, _ = nav.π5_attract_anomaly("event_pattern", top_n=50)

        # Find E0 in results
        e0_poly = None
        for p in anomalies:
            if p.primary_key == "E0":
                e0_poly = p
                break

        assert e0_poly is not None, "E0 should be in anomaly results"
        assert e0_poly.delta_norm > 0.0, f"E0 delta_norm should be > 0, got {e0_poly.delta_norm}"

        session.close()
