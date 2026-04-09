# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from datetime import UTC, datetime

import numpy as np
import pytest
from hypertopos.model.manifest import Contract, Manifest
from hypertopos.model.objects import Edge, Point, Polygon, Solid, SolidSlice
from hypertopos.model.sphere import (
    ColumnSchema,
    Line,
    PartitionConfig,
    Pattern,
    RelationDef,
    Sphere,
)


def test_point_fields():
    p = Point(
        primary_key="CUST-001",
        line_id="customers",
        version=1,
        status="active",
        properties={"name": "Alice", "region": "EMEA"},
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        changed_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert p.primary_key == "CUST-001"
    assert p.status == "active"


def test_edge_directions():
    e_in = Edge(line_id="customers", point_key="CUST-001", status="alive", direction="in")
    Edge(line_id="orders", point_key="ORD-001", status="alive", direction="out")
    e_self = Edge(line_id="sales_val", point_key="VAL-001", status="alive", direction="self")
    assert e_in.direction == "in"
    assert e_self.direction == "self"


def test_edge_is_alive():
    alive = Edge(line_id="l", point_key="p", status="alive", direction="in")
    dead = Edge(line_id="l", point_key="p", status="dead", direction="in")
    assert alive.is_alive() is True
    assert dead.is_alive() is False


def _make_edges():
    return [
        Edge(line_id="customers", point_key="CUST-001", status="alive", direction="in"),
        Edge(line_id="products", point_key="PROD-007", status="alive", direction="in"),
        Edge(line_id="sales_val", point_key="VAL-001", status="dead", direction="self"),
    ]


def test_polygon_is_fact():
    edges = _make_edges()
    p = Polygon(
        primary_key="SALE-001",
        pattern_id="sale_pattern",
        pattern_ver=1,
        pattern_type="event",
        scale=1,
        delta=np.zeros(3),
        delta_norm=0.0,
        is_anomaly=False,
        edges=edges,
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert p.pattern_type == "event"
    assert p.is_event() is True
    assert p.is_anchor() is False


def test_polygon_edges_for_line():
    edges = _make_edges()
    p = Polygon(
        primary_key="SALE-001",
        pattern_id="sale_pattern",
        pattern_ver=1,
        pattern_type="event",
        scale=1,
        delta=np.zeros(3),
        delta_norm=0.0,
        is_anomaly=False,
        edges=edges,
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    cust_edges = p.edges_for_line("customers")
    assert len(cust_edges) == 1
    assert cust_edges[0].point_key == "CUST-001"


def test_polygon_alive_edges():
    edges = _make_edges()
    p = Polygon(
        primary_key="SALE-001",
        pattern_id="sale_pattern",
        pattern_ver=1,
        pattern_type="event",
        scale=1,
        delta=np.zeros(3),
        delta_norm=0.0,
        is_anomaly=False,
        edges=edges,
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    alive = p.alive_edges()
    assert len(alive) == 2


def _make_base_polygon():
    return Polygon(
        primary_key="CUST-001",
        pattern_id="customer_pattern",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(3),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def _make_solid(base_polygon):
    slices = [
        SolidSlice(
            slice_index=0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            deformation_type="structural",
            delta_snapshot=np.array([0.1, 0.2, 0.0]),
            delta_norm_snapshot=0.22,
            pattern_ver=1,
            changed_property=None,
            changed_line_id="customers",
            added_edge=None,
        ),
        SolidSlice(
            slice_index=1,
            timestamp=datetime(2024, 6, 1, tzinfo=UTC),
            deformation_type="edge",
            delta_snapshot=np.array([0.0, 0.2, 0.3]),
            delta_norm_snapshot=0.36,
            pattern_ver=1,
            changed_property=None,
            changed_line_id="products",
            added_edge=None,
        ),
    ]
    return Solid(
        primary_key="CUST-001",
        pattern_id="customer_pattern",
        base_polygon=base_polygon,
        slices=slices,
    )


def test_solid_slice_at_exact():
    solid = _make_solid(_make_base_polygon())
    s = solid.slice_at(datetime(2024, 6, 1, tzinfo=UTC))
    assert s is not None
    assert s.slice_index == 1


def test_solid_slice_at_before_first():
    solid = _make_solid(_make_base_polygon())
    s = solid.slice_at(datetime(2023, 1, 1, tzinfo=UTC))
    assert s is None


def test_solid_slice_at_requires_sorted_slices():
    """slice_at() uses bisect — slices MUST be in timestamp order.
    This test documents the invariant: Solid receives pre-sorted slices."""
    ts1 = datetime(2022, 1, 1, tzinfo=UTC)
    ts2 = datetime(2023, 1, 1, tzinfo=UTC)
    ts3 = datetime(2024, 1, 1, tzinfo=UTC)

    delta = np.zeros(2, dtype=np.float32)

    def make_slice(idx, ts):
        return SolidSlice(
            slice_index=idx,
            timestamp=ts,
            deformation_type="internal",
            delta_snapshot=delta,
            delta_norm_snapshot=0.0,
            pattern_ver=1,
            changed_property=None,
            changed_line_id=None,
            added_edge=None,
        )

    # Slices in chronological order (as build_solid now guarantees)
    slices = [make_slice(2, ts1), make_slice(0, ts2), make_slice(1, ts3)]
    now = datetime(2024, 1, 1, tzinfo=UTC)
    base = Polygon(
        primary_key="K",
        pattern_id="p",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=delta,
        delta_norm=0.0,
        is_anomaly=False,
        edges=[],
        last_refresh_at=now,
        updated_at=now,
    )
    solid = Solid(primary_key="K", pattern_id="p", base_polygon=base, slices=slices)

    # slice_at must find the correct slice regardless of slice_index values
    result = solid.slice_at(datetime(2022, 6, 1, tzinfo=UTC))
    assert result is not None
    assert result.timestamp == ts1

    result2 = solid.slice_at(datetime(2023, 6, 1, tzinfo=UTC))
    assert result2 is not None
    assert result2.timestamp == ts2


def test_pattern_relation_count():
    relations = [
        RelationDef(line_id="customers", direction="in", required=True),
        RelationDef(line_id="products", direction="in", required=True),
        RelationDef(line_id="stores", direction="in", required=False),
    ]
    p = Pattern(
        pattern_id="customer_pattern",
        entity_type="customer",
        pattern_type="anchor",
        relations=relations,
        mu=np.zeros(3),
        sigma_diag=np.ones(3),
        theta=np.full(3, 2.0),
        population_size=1000,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    assert len(p.relations) == 3
    assert p.delta_dim() == 3


def test_line_has_versions():
    line = Line(
        line_id="customers",
        entity_type="customer",
        line_role="anchor",
        pattern_id="customer_pattern",
        partitioning=PartitionConfig(mode="static", columns=["region"]),
        versions=[1, 2],
    )
    assert 1 in line.versions
    assert line.current_version() == 2


def test_line_columns_default_none():
    line = Line(
        line_id="customers",
        entity_type="customer",
        line_role="anchor",
        pattern_id="customer_pattern",
        partitioning=PartitionConfig(mode="static", columns=[]),
        versions=[1],
    )
    assert line.columns is None


def test_line_columns_with_schema():
    cols = [
        ColumnSchema(name="primary_key", type="string"),
        ColumnSchema(name="region", type="string"),
    ]
    line = Line(
        line_id="customers",
        entity_type="customer",
        line_role="anchor",
        pattern_id="customer_pattern",
        partitioning=PartitionConfig(mode="static", columns=[]),
        versions=[1],
        columns=cols,
    )
    assert line.columns is not None
    assert len(line.columns) == 2
    assert line.columns[0].name == "primary_key"
    assert line.columns[1].type == "string"


def test_manifest_line_version():
    m = Manifest(
        manifest_id="m-001",
        agent_id="agent-001",
        snapshot_time=datetime(2024, 1, 1, tzinfo=UTC),
        status="active",
        line_versions={"customers": 2, "products": 1},
        pattern_versions={"customer_pattern": 3},
        alias_versions={},
    )
    assert m.line_version("customers") == 2
    assert m.line_version("unknown") is None


def test_contract_has_pattern():
    c = Contract(manifest_id="m-001", pattern_ids=["customer_pattern", "sale_pattern"])
    assert c.has_pattern("customer_pattern") is True
    assert c.has_pattern("other") is False


def test_sphere_reverse_index():
    relations_sale = [
        RelationDef(line_id="customers", direction="in", required=True),
        RelationDef(line_id="products", direction="in", required=True),
        RelationDef(line_id="sale_value", direction="self", required=True),
    ]
    sale_pattern = Pattern(
        pattern_id="sale_pattern",
        entity_type="sale",
        pattern_type="event",
        relations=relations_sale,
        mu=np.zeros(3),
        sigma_diag=np.ones(3),
        theta=np.full(3, 2.0),
        population_size=50,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    sphere = Sphere(
        sphere_id="test",
        name="Test",
        base_path="/tmp",
        patterns={"sale_pattern": sale_pattern},
    )
    assert "customers" in sphere.reverse_index
    assert "sale_pattern" in sphere.reverse_index["customers"]
    assert "products" in sphere.reverse_index
    assert "sale_pattern" in sphere.reverse_index["products"]


# --- CuttingPlane ---


def test_cutting_plane_contains():
    from hypertopos.model.sphere import CuttingPlane

    cp = CuttingPlane(normal=[1.0, 1.0], bias=1.5)
    assert cp.contains(np.array([1.0, 1.0])) is True  # 2.0 >= 1.5
    assert cp.contains(np.array([0.5, 0.5])) is False  # 1.0 < 1.5
    assert cp.contains(np.array([0.8, 0.7])) is True  # 1.5 >= 1.5 (boundary)


def test_cutting_plane_signed_distance():
    from hypertopos.model.sphere import CuttingPlane

    cp = CuttingPlane(normal=[1.0, 0.0], bias=0.5)
    # w=[1,0], b=0.5, ||w||=1 → signed_dist = delta[0] - 0.5
    assert abs(cp.signed_distance(np.array([1.0, 0.0])) - 0.5) < 1e-5
    assert abs(cp.signed_distance(np.array([0.0, 0.0])) - (-0.5)) < 1e-5


def test_alias_filter_cutting_plane_optional():
    from hypertopos.model.sphere import AliasFilter, CuttingPlane

    # Without cutting_plane — backward compatible
    f = AliasFilter(include_relations=["orders"])
    assert f.cutting_plane is None

    # With cutting_plane
    cp = CuttingPlane(normal=[1.0], bias=0.5)
    f2 = AliasFilter(include_relations=["orders"], cutting_plane=cp)
    assert f2.cutting_plane is not None
    assert f2.cutting_plane.bias == 0.5


def _make_pattern_for_model_tests() -> Pattern:
    return Pattern(
        pattern_id="p1",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(
                line_id="products", direction="out", required=True, display_name="Products"
            ),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.array([3.0, 4.0], dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )


def test_pattern_theta_norm():
    p = _make_pattern_for_model_tests()
    assert p.theta_norm == pytest.approx(5.0, rel=1e-5)


def test_pattern_dim_index_by_line_id():
    p = _make_pattern_for_model_tests()
    assert p.dim_index("products") == 0
    assert p.dim_index("stores") == 1


def test_pattern_dim_index_by_display_name():
    p = _make_pattern_for_model_tests()
    assert p.dim_index("Products") == 0


def test_pattern_dim_index_unknown_raises():
    p = _make_pattern_for_model_tests()
    with pytest.raises(ValueError, match="Dimension 'unknown'"):
        p.dim_index("unknown")


def _make_sphere_for_tests() -> Sphere:
    pattern = _make_pattern_for_model_tests()
    anchor_line = Line(
        line_id="customers",
        entity_type="customer",
        line_role="anchor",
        pattern_id="p1",
        partitioning=PartitionConfig(mode="static", columns=[]),
        versions=[1],
    )
    event_line = Line(
        line_id="sales",
        entity_type="sale",
        line_role="event",
        pattern_id="sale_pattern",
        partitioning=PartitionConfig(mode="static", columns=[]),
        versions=[1],
    )
    return Sphere(
        sphere_id="test",
        name="Test",
        base_path="/tmp/test",
        lines={"customers": anchor_line, "sales": event_line},
        patterns={"p1": pattern},
    )


def test_sphere_entity_line_returns_anchor_line():
    s = _make_sphere_for_tests()
    assert s.entity_line("p1") == "customers"


def test_sphere_entity_line_unknown_pattern_returns_none():
    s = _make_sphere_for_tests()
    assert s.entity_line("nonexistent") is None


def test_sphere_entity_line_event_pattern_returns_none():
    s = _make_sphere_for_tests()
    assert s.entity_line("sale_pattern") is None


def test_pattern_prop_columns_default_empty():
    import datetime

    import numpy as np
    from hypertopos.model.sphere import Pattern, RelationDef

    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[RelationDef(line_id="orders", direction="out", required=True)],
        mu=np.array([0.5], dtype=np.float32),
        sigma_diag=np.array([0.3], dtype=np.float32),
        theta=np.array([1.5], dtype=np.float32),
        population_size=100,
        computed_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        version=1,
        status="production",
    )
    assert pattern.prop_columns == []
    assert pattern.excluded_properties == []


def test_pattern_delta_dim_includes_prop_columns():
    import datetime

    import numpy as np
    from hypertopos.model.sphere import Pattern, RelationDef

    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[RelationDef(line_id="orders", direction="out", required=True)],
        mu=np.array([0.5, 0.9, 0.3], dtype=np.float32),
        sigma_diag=np.array([0.3, 0.1, 0.4], dtype=np.float32),
        theta=np.array([1.5, 2.0, 2.0], dtype=np.float32),
        population_size=100,
        computed_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        version=1,
        status="production",
        prop_columns=["name", "region"],
    )
    assert pattern.delta_dim() == 3  # 1 relation + 2 props


# --- dim_index covers prop_columns ---


def test_pattern_dim_index_prop_column():
    import datetime

    import numpy as np
    from hypertopos.model.sphere import Pattern, RelationDef

    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[RelationDef(line_id="orders", direction="out", required=True)],
        mu=np.array([0.5, 0.9], dtype=np.float32),
        sigma_diag=np.array([0.3, 0.1], dtype=np.float32),
        theta=np.array([1.5, 2.0], dtype=np.float32),
        population_size=100,
        computed_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        version=1,
        status="production",
        prop_columns=["country"],
    )
    assert pattern.dim_index("orders") == 0
    assert pattern.dim_index("country") == 1  # K=1, prop index 0 → K+0=1


def test_pattern_dim_index_unknown_lists_prop_columns():
    import datetime

    import numpy as np
    from hypertopos.model.sphere import Pattern, RelationDef

    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[RelationDef(line_id="orders", direction="out", required=True)],
        mu=np.array([0.5, 0.9], dtype=np.float32),
        sigma_diag=np.array([0.3, 0.1], dtype=np.float32),
        theta=np.array([1.5, 2.0], dtype=np.float32),
        population_size=100,
        computed_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        version=1,
        status="production",
        prop_columns=["country"],
    )
    with pytest.raises(ValueError, match="country"):
        pattern.dim_index("unknown")


# --- Polygon.count_alive_edges_to ---


def test_polygon_count_alive_edges_to():
    edges = [
        Edge(line_id="customers", point_key="C1", status="alive", direction="in"),
        Edge(line_id="customers", point_key="C2", status="alive", direction="in"),
        Edge(line_id="customers", point_key="C3", status="dead", direction="in"),
        Edge(line_id="products", point_key="P1", status="alive", direction="in"),
    ]
    now = datetime(2024, 1, 1, tzinfo=UTC)
    p = Polygon(
        primary_key="S1",
        pattern_id="sp",
        pattern_ver=1,
        pattern_type="event",
        scale=1,
        delta=np.zeros(3),
        delta_norm=0.0,
        is_anomaly=False,
        edges=edges,
        last_refresh_at=now,
        updated_at=now,
    )
    assert p.count_alive_edges_to("customers") == 2
    assert p.count_alive_edges_to("products") == 1
    assert p.count_alive_edges_to("nonexistent") == 0


# --- SolidSlice.prop_column_states ---


def test_solid_slice_prop_column_states():
    # Pattern: 2 relations + 2 prop_columns = 4 dims
    # mu =    [0, 0, 0.9, 0.1]   (prop "active" mu=0.9, "vip" mu=0.1)
    # sigma = [1, 1, 0.2, 0.3]
    # delta_snapshot z-scores: we set prop dims such that inverse z gives >0.5 and <0.5
    # For "active": z * sigma + mu = 0.0 * 0.2 + 0.9 = 0.9 > 0.5 → True
    # For "vip":    z * sigma + mu = 0.0 * 0.3 + 0.1 = 0.1 < 0.5 → False
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=True),
        ],
        mu=np.array([0.0, 0.0, 0.9, 0.1], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0, 0.2, 0.3], dtype=np.float32),
        theta=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        prop_columns=["active", "vip"],
    )
    ss = SolidSlice(
        slice_index=0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        deformation_type="internal",
        delta_snapshot=np.array([0.5, -0.3, 0.0, 0.0], dtype=np.float32),
        delta_norm_snapshot=0.58,
        pattern_ver=1,
        changed_property=None,
        changed_line_id=None,
        added_edge=None,
    )
    states = ss.prop_column_states(pattern)
    assert states == {"active": True, "vip": False}


# --- SolidSlice.delta_relations ---


def test_solid_slice_delta_relations():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=True),
        ],
        mu=np.array([0.0, 0.0, 0.9], dtype=np.float32),
        sigma_diag=np.array([1.0, 1.0, 0.2], dtype=np.float32),
        theta=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        prop_columns=["active"],
    )
    ss = SolidSlice(
        slice_index=0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        deformation_type="internal",
        delta_snapshot=np.array([0.1234, -0.5678, 1.0], dtype=np.float32),
        delta_norm_snapshot=0.58,
        pattern_ver=1,
        changed_property=None,
        changed_line_id=None,
        added_edge=None,
    )
    rels = ss.delta_relations(pattern)
    assert len(rels) == 2
    assert rels[0] == pytest.approx(0.1234, abs=1e-4)
    assert rels[1] == pytest.approx(-0.5678, abs=1e-4)


# --- Pattern.dim_labels ---


def test_pattern_dim_labels():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True, display_name="Orders"),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        sigma_diag=np.ones(3, dtype=np.float32),
        theta=np.ones(3, dtype=np.float32),
        population_size=50,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        prop_columns=["region"],
    )
    labels = pattern.dim_labels
    assert labels == ["Orders", "stores", "region"]


# --- Pattern.max_hub_score ---


def test_pattern_max_hub_score_with_edge_max():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
            RelationDef(line_id="stores", direction="in", required=False),
        ],
        mu=np.zeros(2, dtype=np.float32),
        sigma_diag=np.ones(2, dtype=np.float32),
        theta=np.ones(2, dtype=np.float32),
        population_size=50,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=np.array([10.0, 5.0], dtype=np.float32),
    )
    assert pattern.max_hub_score == pytest.approx(15.0)


def test_pattern_max_hub_score_without_edge_max():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=50,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    assert pattern.max_hub_score is None


# --- Pattern.is_continuous ---


def test_pattern_is_continuous_true():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=50,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        edge_max=np.array([10.0], dtype=np.float32),
    )
    assert pattern.is_continuous is True


def test_pattern_is_continuous_false():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=50,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    assert pattern.is_continuous is False


# --- Pattern.effective_sample_size ---


def test_pattern_effective_sample_size():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=1000,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    assert pattern.effective_sample_size(0.1) == 100
    assert pattern.effective_sample_size(0.5) == 500
    assert pattern.effective_sample_size(0.001) == 1  # max(1, 1000*0.001=1)


def test_pattern_effective_sample_size_minimum_one():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=10,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    # 10 * 0.001 = 0.01 → int(0.01) = 0 → max(1, 0) = 1
    assert pattern.effective_sample_size(0.001) == 1


# --- Sphere.sibling_lines ---


def _make_line(line_id: str, source_id: str | None = None) -> Line:
    return Line(
        line_id=line_id,
        entity_type="account",
        line_role="anchor",
        pattern_id=f"{line_id}_pattern",
        partitioning=PartitionConfig(mode="static", columns=[]),
        versions=[1],
        source_id=source_id,
    )


def _make_sphere_with_siblings() -> Sphere:
    pattern = _make_pattern_for_model_tests()
    accounts = _make_line("accounts", source_id="src_accounts")
    accounts_stress = _make_line("accounts_stress", source_id="src_accounts")
    customers = _make_line("customers", source_id=None)
    return Sphere(
        sphere_id="test",
        name="Test",
        base_path="/tmp/test",
        lines={
            "accounts": accounts,
            "accounts_stress": accounts_stress,
            "customers": customers,
        },
        patterns={"p1": pattern},
    )


def test_sibling_lines_returns_siblings():
    s = _make_sphere_with_siblings()
    assert s.sibling_lines("accounts") == ["accounts_stress"]
    assert s.sibling_lines("accounts_stress") == ["accounts"]


def test_sibling_lines_no_source_id_returns_empty():
    s = _make_sphere_with_siblings()
    assert s.sibling_lines("customers") == []


def test_sibling_lines_unknown_line_returns_empty():
    s = _make_sphere_with_siblings()
    assert s.sibling_lines("nonexistent") == []


def test_sibling_lines_three_siblings():
    pattern = _make_pattern_for_model_tests()
    a = _make_line("a", source_id="shared")
    b = _make_line("b", source_id="shared")
    c = _make_line("c", source_id="shared")
    d = _make_line("d", source_id=None)
    s = Sphere(
        sphere_id="test",
        name="Test",
        base_path="/tmp/test",
        lines={"a": a, "b": b, "c": c, "d": d},
        patterns={"p1": pattern},
    )
    siblings_of_a = s.sibling_lines("a")
    assert set(siblings_of_a) == {"b", "c"}
    assert "a" not in siblings_of_a


# --- Pattern.dim_percentiles ---


def test_pattern_dim_percentiles_default_none():
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
    )
    assert pattern.dim_percentiles is None


def test_pattern_dim_percentiles_stored():
    percentiles = {
        "avg_late_days": {"min": 0.0, "p25": 20.0, "p50": 26.0, "p75": 32.0, "p99": 50.0, "max": 322.0},
    }
    pattern = Pattern(
        pattern_id="p",
        entity_type="customer",
        pattern_type="anchor",
        relations=[
            RelationDef(line_id="orders", direction="out", required=True),
        ],
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32),
        population_size=100,
        computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        version=1,
        status="production",
        dim_percentiles=percentiles,
    )
    assert pattern.dim_percentiles == percentiles
    assert pattern.dim_percentiles["avg_late_days"]["max"] == 322.0
