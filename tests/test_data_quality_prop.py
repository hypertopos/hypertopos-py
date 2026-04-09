# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Test that detect_data_quality_issues flags zero-variance prop columns."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
from hypertopos.model.manifest import Manifest
from hypertopos.model.sphere import Pattern, RelationDef
from hypertopos.navigation.navigator import GDSNavigator

# ---------------------------------------------------------------------------
# Helpers — reusable pattern / navigator factories
# ---------------------------------------------------------------------------


def _make_pattern(
    pid: str,
    prop_columns: list[str] | None = None,
    extra_sigma: list[float] | None = None,
) -> Pattern:
    """Pattern with one relation; optionally adds prop_columns with custom sigma."""
    rel = RelationDef(
        line_id="products",
        direction="in",
        required=True,
        display_name="products",
    )
    base_sigma = [0.2]  # one relation dimension
    if extra_sigma:
        base_sigma.extend(extra_sigma)

    base_mu = [0.5]
    if extra_sigma:
        base_mu.extend([0.0] * len(extra_sigma))

    base_theta = [1.5]
    if extra_sigma:
        base_theta.extend([1.0] * len(extra_sigma))

    return Pattern(
        pattern_id=pid,
        entity_type="customers",
        pattern_type="anchor",
        relations=[rel],
        mu=np.array(base_mu, dtype=np.float32),
        sigma_diag=np.array(base_sigma, dtype=np.float32),
        theta=np.array(base_theta, dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=None,
        prop_columns=prop_columns or [],
        excluded_properties=[],
    )


def _make_dqi_navigator(
    pattern: Pattern,
    geo_rows: int = 10,
    prop_dim_values: dict[int, list[float]] | None = None,
) -> GDSNavigator:
    """Navigator mocked for detect_data_quality_issues.

    prop_dim_values: {dim_index: [values_per_row]} — controls delta values on
    specific dimensions. Values must have length == geo_rows. Dimensions not in
    this dict default to 1.0 on dim 0, 0.0 elsewhere.
    """
    storage = MagicMock()
    engine = MagicMock()
    cache = MagicMock()

    sphere = MagicMock()
    sphere.patterns = {pattern.pattern_id: pattern}
    sphere.lines = {}

    storage.read_sphere.return_value = sphere
    storage.read_geometry_stats.return_value = None
    storage.count_geometry_rows.side_effect = lambda pid, ver, filter=None: (
        0 if filter else geo_rows
    )

    manifest = Manifest(
        manifest_id=str(uuid.uuid4()),
        agent_id="test",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 1},
        pattern_versions={pattern.pattern_id: 1},
    )

    # Build edges struct table (all entities have "products" alive)
    from hypertopos.storage._schemas import EDGE_STRUCT_TYPE

    n = geo_rows
    edges_data = [
        [{"line_id": "products", "point_key": f"P-{i}", "status": "alive", "direction": "in"}]
        for i in range(n)
    ]
    eli_table = pa.table(
        {
            "edges": pa.array(edges_data, type=pa.list_(EDGE_STRUCT_TYPE)),
        }
    )

    # Delta table — with controllable per-dimension values
    dim = len(pattern.sigma_diag)
    base_delta = [1.0] + [0.0] * (dim - 1) if dim > 1 else [1.0]
    deltas = [list(base_delta) for _ in range(n)]
    if prop_dim_values:
        for d_idx, vals in prop_dim_values.items():
            for row_i, v in enumerate(vals):
                deltas[row_i][d_idx] = v
    delta_norms = [float(np.linalg.norm(d)) for d in deltas]

    consistent_table = pa.table(
        {
            "primary_key": pa.array([f"E-{i}" for i in range(n)]),
            "delta": pa.array(deltas, type=pa.list_(pa.float32())),
            "delta_norm": pa.array(delta_norms, type=pa.float32()),
        }
    )

    # read_geometry for prop column check (returns delta column)
    def _read_geometry(pid, ver, columns=None):
        if columns and "delta" in columns:
            return pa.table(
                {
                    "delta": pa.array(deltas, type=pa.list_(pa.float32())),
                }
            )
        return consistent_table

    storage.read_geometry.side_effect = _read_geometry

    call_count = [0]

    def _batched(pid, ver, columns=None, filter_expr=None, batch_size=65_536):
        call_count[0] += 1
        if call_count[0] == 1:
            return iter(eli_table.to_batches())
        return iter(consistent_table.to_batches())

    storage.read_geometry_batched.side_effect = _batched

    nav = GDSNavigator.__new__(GDSNavigator)
    nav._storage = storage
    nav._engine = engine
    nav._cache = cache
    nav._manifest = manifest
    return nav


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPropColumnZeroVariance:
    def test_zero_variance_prop_flagged(self):
        """Prop column with zero delta variance should produce MEDIUM finding.

        sigma=0.2 (SIGMA_EPS_PROP floor) but all deltas are constant on the
        prop dim → delta variance = 0.0 → must be flagged.
        """
        n = 10
        pattern = _make_pattern(
            "customer_pattern",
            prop_columns=["dead_prop"],
            extra_sigma=[0.2],  # sigma floored at 0.2, but deltas are constant
        )
        # All entities have identical delta on prop dim (dim 1) = 0.0
        nav = _make_dqi_navigator(
            pattern,
            geo_rows=n,
            prop_dim_values={1: [0.0] * n},
        )

        findings = nav.detect_data_quality_issues("customer_pattern")
        prop_findings = [f for f in findings if f["issue_type"] == "zero_variance_prop_column"]
        assert len(prop_findings) == 1
        assert prop_findings[0]["dimension"] == "dead_prop"
        assert prop_findings[0]["severity"] == "MEDIUM"
        assert prop_findings[0]["delta_variance"] < 0.01

    def test_normal_variance_prop_not_flagged(self):
        """Prop column with real delta variance should NOT be flagged."""
        n = 10
        pattern = _make_pattern(
            "customer_pattern",
            prop_columns=["good_prop"],
            extra_sigma=[0.5],
        )
        # Deltas vary significantly on prop dim
        rng = np.random.default_rng(42)
        varying_vals = rng.normal(0.0, 1.0, size=n).tolist()
        nav = _make_dqi_navigator(
            pattern,
            geo_rows=n,
            prop_dim_values={1: varying_vals},
        )

        findings = nav.detect_data_quality_issues("customer_pattern")
        prop_findings = [f for f in findings if f["issue_type"] == "zero_variance_prop_column"]
        assert len(prop_findings) == 0

    def test_no_prop_columns_no_finding(self):
        """Pattern without prop_columns should not produce prop findings."""
        pattern = _make_pattern("customer_pattern")
        nav = _make_dqi_navigator(pattern)

        findings = nav.detect_data_quality_issues("customer_pattern")
        prop_findings = [f for f in findings if f["issue_type"] == "zero_variance_prop_column"]
        assert len(prop_findings) == 0

    def test_multiple_prop_columns_mixed(self):
        """Only zero-variance props should be flagged when mixed."""
        n = 10
        pattern = _make_pattern(
            "customer_pattern",
            prop_columns=["bad_prop", "good_prop"],
            extra_sigma=[0.2, 0.3],
        )
        rng = np.random.default_rng(42)
        nav = _make_dqi_navigator(
            pattern,
            geo_rows=n,
            prop_dim_values={
                1: [0.0] * n,  # bad_prop: constant → zero variance
                2: rng.normal(0.0, 1.0, size=n).tolist(),  # good_prop: varies
            },
        )

        findings = nav.detect_data_quality_issues("customer_pattern")
        prop_findings = [f for f in findings if f["issue_type"] == "zero_variance_prop_column"]
        assert len(prop_findings) == 1
        assert prop_findings[0]["dimension"] == "bad_prop"

    def test_finding_contains_dim_index(self):
        """Finding should report the correct dim_index (n_rel + j)."""
        n = 10
        pattern = _make_pattern(
            "customer_pattern",
            prop_columns=["low_var_prop"],
            extra_sigma=[0.2],
        )
        nav = _make_dqi_navigator(
            pattern,
            geo_rows=n,
            prop_dim_values={1: [0.0] * n},
        )

        findings = nav.detect_data_quality_issues("customer_pattern")
        prop_findings = [f for f in findings if f["issue_type"] == "zero_variance_prop_column"]
        assert len(prop_findings) == 1
        # 1 relation + index 0 in prop_columns → dim_index = 1
        assert prop_findings[0]["dim_index"] == 1

    def test_sigma_floored_but_dead_detected(self):
        """Key regression: sigma=0.2 (floor) must NOT prevent detection.

        This is the exact bug from the stress test: sigma is floored at
        SIGMA_EPS_PROP=0.2 so the old check (sigma < 0.01) never fired.
        The new check measures actual delta variance instead.
        """
        n = 20
        pattern = _make_pattern(
            "customer_pattern",
            prop_columns=["fashion_news_frequency"],
            extra_sigma=[0.2],  # exactly SIGMA_EPS_PROP
        )
        nav = _make_dqi_navigator(
            pattern,
            geo_rows=n,
            prop_dim_values={1: [0.0] * n},  # all zeros despite sigma=0.2
        )

        findings = nav.detect_data_quality_issues("customer_pattern")
        prop_findings = [f for f in findings if f["issue_type"] == "zero_variance_prop_column"]
        assert len(prop_findings) == 1, (
            "sigma=0.2 (SIGMA_EPS_PROP floor) with zero delta variance "
            "must be detected — old sigma-based check missed this"
        )
