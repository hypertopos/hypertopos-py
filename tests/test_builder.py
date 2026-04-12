# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypertopos.builder import GDSBuilder, RelationSpec


def test_builder_init(tmp_path):
    b = GDSBuilder("test_sphere", str(tmp_path / "gds_test"))
    assert b.sphere_id == "test_sphere"


def test_add_line_registers(tmp_path):
    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", [{"customer_id": "C-1", "name": "Acme"}], key_col="customer_id", source_id="test")
    assert "customers" in b._lines


def test_add_pattern_registers(tmp_path):
    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", [{"customer_id": "C-1"}], key_col="customer_id", source_id="test")
    b.add_line("orders", [{"order_id": "O-1", "customer_id": "C-1"}], key_col="order_id", source_id="test")
    b.add_pattern(
        "order_pattern",
        pattern_type="event",
        entity_line="orders",
        relations=[RelationSpec("customers", fk_col="customer_id", direction="in", required=True)],
    )
    assert "order_pattern" in b._patterns


def test_add_line_normalizes_key_col(tmp_path):
    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", [{"customer_id": "C-1", "name": "Acme"}], key_col="customer_id", source_id="test")
    tbl = b._lines["customers"].table
    assert "primary_key" in tbl.schema.names
    assert "version" in tbl.schema.names
    assert "status" in tbl.schema.names
    assert "created_at" in tbl.schema.names
    assert tbl["primary_key"][0].as_py() == "C-1"
    assert tbl["status"][0].as_py() == "active"


def test_add_line_accepts_arrow_table(tmp_path):
    b = GDSBuilder("s", str(tmp_path / "gds"))
    tbl = pa.table({"id": ["X-1"], "val": [42]})
    b.add_line("items", tbl, key_col="id", source_id="test")
    assert "primary_key" in b._lines["items"].table.schema.names


def test_add_line_returns_self_for_chaining(tmp_path):
    b = GDSBuilder("s", str(tmp_path / "gds"))
    result = b.add_line("x", [{"id": "1"}], key_col="id", source_id="test")
    assert result is b


def _make_minimal_builder(tmp_path):
    """Helper: build a minimal sphere with 5 customers, 2 company_codes, 8 gl_entries."""
    b = GDSBuilder("test", str(tmp_path / "gds_test"))
    b.add_line(
        "customers",
        [
            {"cust_id": "C-1", "name": "Alpha"},
            {"cust_id": "C-2", "name": "Beta"},
            {"cust_id": "C-3", "name": "Gamma"},
            {"cust_id": "C-4", "name": "Delta"},
            {"cust_id": "C-5", "name": "Epsilon"},
        ],
        key_col="cust_id",
        source_id="test",
    )
    b.add_line(
        "company_codes",
        [
            {"cc_id": "CC-01", "name": "Germany"},
            {"cc_id": "CC-02", "name": "USA"},
        ],
        key_col="cc_id",
        source_id="test",
    )
    b.add_line(
        "gl_entries",
        [
            {"doc": "D-001", "cust_id": "C-1", "cc_id": "CC-01"},
            {"doc": "D-002", "cust_id": "C-1", "cc_id": "CC-01"},
            {"doc": "D-003", "cust_id": "C-2", "cc_id": "CC-02"},
            {"doc": "D-004", "cust_id": "C-3", "cc_id": "CC-01"},
            {"doc": "D-005", "cust_id": "C-4", "cc_id": "CC-01"},
            {"doc": "D-006", "cust_id": None, "cc_id": "CC-01"},
            {"doc": "D-007", "cust_id": None, "cc_id": "CC-02"},
            {"doc": "D-008", "cust_id": None, "cc_id": None},
        ],
        key_col="doc",
        source_id="test",
        role="event",
    )
    b.add_pattern(
        "gl_entry_pattern",
        pattern_type="event",
        entity_line="gl_entries",
        relations=[
            RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
            RelationSpec("company_codes", fk_col="cc_id", direction="in", required=True),
        ],
        anomaly_percentile=60.0,
    )
    return b


def test_add_line_list_uses_union_of_all_row_keys(tmp_path):
    """Columns from rows other than row[0] must not be silently dropped."""
    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "items",
        [
            {"id": "X-1", "name": "Alpha"},
            {"id": "X-2", "name": "Beta", "extra": "value"},
        ],
        key_col="id",
        source_id="test",
    )
    assert "extra" in b._lines["items"].table.schema.names


def test_validate_rejects_fk_col_none_with_non_self_direction(tmp_path):
    """fk_col=None with direction!='self' must raise on build(), not crash silently."""
    import pytest

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", [{"id": "C-1"}], key_col="id", source_id="test")
    b.add_line("orders", [{"oid": "O-1"}], key_col="oid", source_id="test")
    b.add_pattern(
        "op",
        pattern_type="event",
        entity_line="orders",
        relations=[RelationSpec("customers", fk_col=None, direction="in")],
    )
    with pytest.raises(ValueError, match="fk_col must not be None"):
        b.build()


def test_build_geometry_table(tmp_path):
    b = _make_minimal_builder(tmp_path)
    pat = b._patterns["gl_entry_pattern"]
    result = b._build_geometry_table(pat)
    tbl = result[0]
    assert tbl.num_rows == 8
    assert "delta" in tbl.schema.names
    assert "delta_norm" in tbl.schema.names
    assert "is_anomaly" in tbl.schema.names
    # Event patterns: no edges column, only entity_keys
    assert "edges" not in tbl.schema.names
    assert "entity_keys" in tbl.schema.names
    # Verify dead edges via entity_keys: empty string = dead
    for i in range(tbl.num_rows):
        row_key = tbl["primary_key"][i].as_py()
        entity_keys = tbl["entity_keys"][i].as_py() or []
        # First relation is "customers" (index 0)
        cust_key = entity_keys[0] if len(entity_keys) > 0 else ""
        if row_key in ("D-006", "D-007", "D-008"):
            assert cust_key == "", f"Expected dead edge for {row_key}, got '{cust_key}'"


def test_geometry_has_edges_struct_not_json(tmp_path):
    """Geometry table uses native Arrow struct for edges, not JSON string."""
    import lance
    import pyarrow as pa
    from hypertopos.builder.builder import GDSBuilder, RelationSpec

    builder = GDSBuilder("s", str(tmp_path / "gds"))
    customers = pa.table(
        {
            "customer_id": ["C-1", "C-2"],
            "segment_id": ["S-A", "S-B"],
        }
    )
    segments = pa.table({"segment_id": ["S-A", "S-B"]})
    builder.add_line("customers", customers, key_col="customer_id", source_id="test")
    builder.add_line("segments", segments, key_col="segment_id", source_id="test")
    builder.add_pattern(
        "customer_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec(line_id="customers", fk_col=None, direction="self"),
            RelationSpec(line_id="segments", fk_col="segment_id", direction="in"),
        ],
    )
    builder.build()

    geo_path = tmp_path / "gds" / "geometry" / "customer_pattern" / "v=1" / "data.lance"
    geo_table = lance.dataset(str(geo_path)).to_table()

    assert "edges" in geo_table.schema.names
    assert "edges_json" not in geo_table.schema.names
    edges_type = geo_table.schema.field("edges").type
    assert pa.types.is_list(edges_type)
    first_edges = geo_table["edges"][0].as_py()
    assert isinstance(first_edges, list)
    assert isinstance(first_edges[0], dict)
    assert set(first_edges[0].keys()) == {"line_id", "point_key", "status", "direction"}


def test_build_creates_sphere_json(tmp_path):
    import json as json_mod
    from pathlib import Path

    b = _make_minimal_builder(tmp_path)
    out = b.build()
    sphere_json = Path(out) / "_gds_meta" / "sphere.json"
    assert sphere_json.exists()
    data = json_mod.loads(sphere_json.read_text())
    assert data["sphere_id"] == "test"
    assert "customers" in data["lines"]
    assert "gl_entry_pattern" in data["patterns"]
    assert "aliases" in data
    # last_calibrated_at must be written per pattern
    for pid, pat in data["patterns"].items():
        assert "last_calibrated_at" in pat, (
            f"Pattern {pid} missing last_calibrated_at in sphere.json"
        )  # noqa: E501
    # columns must be embedded per line
    for lid, line_data in data["lines"].items():
        assert "columns" in line_data, f"Line {lid} missing columns in sphere.json"
        col_names = [c["name"] for c in line_data["columns"]]
        assert "primary_key" in col_names
        for internal in ("version", "status", "created_at", "changed_at"):
            assert internal not in col_names, f"Internal column {internal} leaked into {lid}"


def test_build_creates_points_files(tmp_path):
    from pathlib import Path

    b = _make_minimal_builder(tmp_path)
    out = b.build()
    for line in ("customers", "gl_entries"):
        v1 = Path(out) / "points" / line / "v=1"
        assert (v1 / "data.lance").exists(), f"No data.lance in {v1}"


def test_build_creates_geometry_file(tmp_path):
    from pathlib import Path

    b = _make_minimal_builder(tmp_path)
    out = b.build()
    assert (Path(out) / "geometry" / "gl_entry_pattern" / "v=1" / "data.lance").exists()


def test_integration_build_and_navigate(tmp_path):
    """Build a minimal sphere, open it with HyperSphere, navigate with GDSNavigator."""
    from hypertopos.sphere import HyperSphere

    b = _make_minimal_builder(tmp_path)
    out = b.build()

    sphere = HyperSphere.open(out)
    assert "gl_entry_pattern" in sphere._sphere.patterns
    assert "customers" in sphere._sphere.lines

    session = sphere.session("test_agent")
    nav = session.navigator()

    # π1 — goto a known entity
    nav.goto("D-001", "gl_entries")
    pos = nav.position
    assert pos is not None
    assert pos.primary_key == "D-001"

    # current_polygon — check delta and edges
    polygon = nav.current_polygon("gl_entry_pattern")
    assert polygon.primary_key == "D-001"
    assert polygon.delta is not None
    assert len(polygon.delta) == 2  # 2 dimensions: customers, company_codes

    # D-001 has both customer and cc → both edges alive
    alive_lines = {e.line_id for e in polygon.alive_edges()}
    assert "customers" in alive_lines
    assert "company_codes" in alive_lines

    # D-006 is anomalous (missing customer)
    nav.goto("D-006", "gl_entries")
    anomalous_poly = nav.current_polygon("gl_entry_pattern")
    assert anomalous_poly.is_anomaly

    # D-001 is normal
    nav.goto("D-001", "gl_entries")
    normal_poly = nav.current_polygon("gl_entry_pattern")
    assert not normal_poly.is_anomaly

    session.close()


def test_integration_find_anomalies(tmp_path):
    from hypertopos.sphere import HyperSphere

    b = _make_minimal_builder(tmp_path)
    out = b.build()
    sphere = HyperSphere.open(out)
    session = sphere.session("test_agent_2")
    nav = session.navigator()

    anomalies, _, _, _ = nav.π5_attract_anomaly("gl_entry_pattern", top_n=10)
    anomaly_keys = {p.primary_key for p in anomalies}
    assert "D-006" in anomaly_keys
    assert "D-008" in anomaly_keys

    session.close()


def test_write_points_unpartitioned(tmp_path):
    from hypertopos.builder._writer import write_points

    tbl = pa.table(
        {
            "primary_key": ["A", "B"],
            "version": pa.array([1, 1], type=pa.int32()),
            "status": ["active", "active"],
            "val": [10, 20],
        }
    )
    write_points(tmp_path, "items", tbl, version=1, partition_col=None)
    v1 = tmp_path / "points" / "items" / "v=1"
    lance_out = v1 / "data.lance"
    parquet_out = v1 / "data.parquet"
    assert lance_out.exists() or parquet_out.exists()
    if lance_out.exists():
        import lance

        result = lance.dataset(str(lance_out)).to_table()
    else:
        result = pq.read_table(str(parquet_out))
    assert result.num_rows == 2
    assert "primary_key" in result.schema.names


def test_compute_stats_basic():
    from hypertopos.builder._stats import compute_stats

    shape_vectors = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],  # missing dim1
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    mu, sigma, theta, _, _, _ = compute_stats(shape_vectors, anomaly_percentile=95.0)
    assert mu.shape == (2,)
    assert sigma.shape == (2,)
    assert theta.shape == (2,)
    assert abs(mu[0] - 1.0) < 1e-4
    assert abs(mu[1] - 0.9) < 1e-4
    # sigma[0] = 0 for constant dim → clamped to SIGMA_EPS
    assert sigma[0] >= 1e-2
    # theta norm reconstructs scalar threshold
    theta_norm = float(np.linalg.norm(theta))
    assert theta_norm > 0


def test_compute_stats_theta_at_percentile():
    from hypertopos.builder._stats import compute_stats

    normal = np.ones((95, 3), dtype=np.float32)
    anomalous = np.zeros((5, 3), dtype=np.float32)
    shape_vectors = np.vstack([normal, anomalous])
    mu, sigma, theta, _, _, _ = compute_stats(shape_vectors, anomaly_percentile=95.0)
    theta_norm = float(np.linalg.norm(theta))
    deltas = (shape_vectors - mu) / np.maximum(sigma, 1e-2)
    norms = np.linalg.norm(deltas, axis=1)
    n_anomalous = int((norms > theta_norm).sum())
    assert n_anomalous <= 6  # allow ±1 for borderline cases


def test_write_points_partitioned(tmp_path):
    from hypertopos.builder._writer import write_points

    tbl = pa.table(
        {
            "primary_key": ["A", "B", "C"],
            "version": pa.array([1, 1, 1], type=pa.int32()),
            "status": ["active", "active", "active"],
            "region": ["EMEA", "EMEA", "APAC"],
        }
    )
    write_points(tmp_path, "customers", tbl, version=1, partition_col="region")
    v1 = tmp_path / "points" / "customers" / "v=1"
    lance_out = v1 / "data.lance"
    if lance_out.exists():
        # Lance: single dataset, no Hive partitions — all rows in one dataset
        import lance

        result = lance.dataset(str(lance_out)).to_table()
        assert result.num_rows == 3
        regions = result["region"].to_pylist()
        assert sorted(regions) == ["APAC", "EMEA", "EMEA"]
    else:
        # Parquet fallback: Hive-partitioned
        emea = v1 / "region=EMEA" / "data.parquet"
        apac = v1 / "region=APAC" / "data.parquet"
        assert emea.exists() and apac.exists()
        assert pq.ParquetFile(str(emea)).read().num_rows == 2
        assert pq.ParquetFile(str(apac)).read().num_rows == 1


def _make_large_builder(tmp_path, n_entities: int = 300):
    """Build a sphere with n_entities GL entries (enough to trigger IVF-PQ index)."""
    b = GDSBuilder("large", str(tmp_path / "gds_large"))
    customers = [{"cust_id": f"C-{i:04d}", "name": f"Customer {i}"} for i in range(n_entities)]
    b.add_line("customers", customers, key_col="cust_id", source_id="test")
    gl_entries = [
        {"doc": f"D-{i:04d}", "cust_id": f"C-{i % n_entities:04d}"} for i in range(n_entities)
    ]
    b.add_line("gl_entries", gl_entries, key_col="doc", source_id="test", role="event")
    b.add_pattern(
        "gl_entry_pattern",
        pattern_type="event",
        entity_line="gl_entries",
        relations=[
            RelationSpec("customers", fk_col="cust_id", direction="in", required=True),
        ],
    )
    return b


def test_builder_geometry_delta_is_fixed_size_list(tmp_path):
    """write_geometry must cast delta to fixed-size list so Lance can index it."""
    import lance

    b = _make_minimal_builder(tmp_path)
    b.build()

    geo_path = str(tmp_path / "gds_test" / "geometry" / "gl_entry_pattern" / "v=1" / "data.lance")
    ds = lance.dataset(geo_path)
    delta_field = ds.schema.field("delta")
    assert hasattr(delta_field.type, "list_size"), (
        "delta column must be a fixed-size list (FixedSizeList) for IVF-PQ indexing; "
        f"got {delta_field.type}"
    )
    assert delta_field.type.list_size == 2  # 2 relations: customers, company_codes


def test_builder_geometry_has_ann_index_when_large(tmp_path):
    """GDSBuilder writes IVF-PQ index on delta when entity count >= 256."""
    import lance

    b = _make_large_builder(tmp_path, n_entities=300)
    b.build()

    geo_path = str(tmp_path / "gds_large" / "geometry" / "gl_entry_pattern" / "v=1" / "data.lance")
    ds = lance.dataset(geo_path)

    # Lance stores index metadata accessible via ds.list_indices()
    indices = ds.list_indices()
    index_names = [idx["name"] if isinstance(idx, dict) else idx.name for idx in indices]
    vector_indices = [
        n
        for n in index_names
        if "delta" in str(n).lower() or "ivf" in str(n).lower() or "vector" in str(n).lower()
    ]
    assert len(vector_indices) > 0, (
        f"Expected an IVF-PQ vector index on delta, got indices: {index_names}"
    )


def test_builder_find_nearest_uses_ann_path(tmp_path):
    """find_nearest_lance uses the ANN path (not NumPy fallback) after GDSBuilder.build()."""
    from hypertopos.storage.reader import GDSReader

    b = _make_large_builder(tmp_path, n_entities=300)
    out = b.build()

    reader = GDSReader(base_path=out)
    ref_delta = np.array([0.5], dtype=np.float32)

    # find_nearest_lance must return a non-None result (ANN path taken)
    ann_result = reader.find_nearest_lance("gl_entry_pattern", 1, ref_delta, k=5)
    assert ann_result is not None, (
        "find_nearest_lance returned None — Lance dataset does not exist or ANN path is broken"
    )
    assert len(ann_result) > 0, "ANN search returned empty results"
    keys, dists = zip(*ann_result, strict=False)
    assert all(d >= 0.0 for d in dists), "All distances must be non-negative"


def test_build_index_if_needed_is_idempotent(tmp_path):
    """build_index_if_needed can be called multiple times without error."""
    from hypertopos.storage.writer import GDSWriter

    b = _make_large_builder(tmp_path, n_entities=300)
    out = b.build()

    writer = GDSWriter(out)
    # Call twice — must not raise
    writer.build_index_if_needed("gl_entry_pattern", version=1)
    writer.build_index_if_needed("gl_entry_pattern", version=1)


def test_build_index_if_needed_noop_on_missing_dataset(tmp_path):
    """build_index_if_needed does nothing when geometry dataset does not exist."""
    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(str(tmp_path))
    # No dataset at all — must not raise
    writer.build_index_if_needed("nonexistent_pattern", version=1)


def test_build_does_not_call_standalone_compaction(tmp_path):
    """GDSBuilder.build() succeeds without standalone compaction module."""
    builder = GDSBuilder("s", str(tmp_path / "gds"))
    customers = pa.table({"customer_id": ["C-1", "C-2"], "name": ["A", "B"]})
    builder.add_line("customers", customers, key_col="customer_id", source_id="test")
    builder.add_pattern(
        "customer_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("customers", fk_col=None, direction="self")],
    )

    # Pre-create temporal dir
    (tmp_path / "gds" / "temporal" / "customer_pattern").mkdir(parents=True)

    builder.build()  # should succeed without error


def test_pattern_with_no_dimensions_raises(tmp_path):
    """A pattern with zero dimensions has nothing to compute geometry from
    and must be rejected up front. Format 2.1+ Lance writes panic on
    fixed_size_list[0] columns; even if they didn't, a zero-dim geometry has
    no semantic meaning."""
    builder = GDSBuilder("s", str(tmp_path / "gds"))
    customers = pa.table({"customer_id": ["C-1", "C-2"], "name": ["A", "B"]})
    builder.add_line("customers", customers, key_col="customer_id", source_id="test")
    builder.add_pattern(
        "naked_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[],
    )
    with pytest.raises(ValueError, match="no dimensions"):
        builder.build()


# ---------------------------------------------------------------------------
# Property completeness calibration in builder
# ---------------------------------------------------------------------------

MIN_FILL_RATE = 0.05  # mirror constant from builder.py


def test_builder_prop_columns_calibrated(tmp_path):
    """Property fill dims appear in pattern mu/sigma and geometry delta vectors."""
    import json

    b = GDSBuilder("s", str(tmp_path / "gds"))
    # Customers with a FK to segments (seg_id) + tracked properties (name, region)
    b.add_line(
        "customers",
        [
            {"cid": "C-1", "seg_id": "SEG-A", "name": "Alpha", "region": "EMEA"},
            {"cid": "C-2", "seg_id": "SEG-A", "name": "Beta", "region": None},
            {"cid": "C-3", "seg_id": "SEG-B", "name": "Gamma", "region": "APAC"},
            {"cid": "C-4", "seg_id": "SEG-A", "name": None, "region": "EMEA"},
            {"cid": "C-5", "seg_id": "SEG-B", "name": "Eps", "region": "APAC"},
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_line("segments", [{"sid": "SEG-A"}, {"sid": "SEG-B"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cust_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("segments", fk_col="seg_id", direction="in", required=True)],
        tracked_properties=["name", "region"],
    )
    b.build()

    sphere_json = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_json["patterns"]["cust_pattern"]

    assert pat["prop_columns"] == ["name", "region"]
    assert pat["excluded_properties"] == []
    assert len(pat["mu"]) == 3  # 1 edge + 2 props
    assert abs(pat["mu"][1] - 0.8) < 0.01  # fill_rate(name) = 4/5 = 0.8
    assert abs(pat["mu"][2] - 0.8) < 0.01  # fill_rate(region) = 4/5 = 0.8


def test_builder_prop_excluded_when_fill_rate_low(tmp_path):
    """Property with fill_rate < MIN or numeric fill_rate >= MAX is excluded."""
    import json

    data = [
        {
            "cid": f"C-{i}",
            "seg_id": "SEG-A",
            "score": 1,
            "mid_col": f"M{i}" if i < 5 else None,  # 50% fill — included
            "rare_col": None,  # 0% fill — excluded (MIN)
        }
        for i in range(10)
    ]
    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", data, key_col="cid", source_id="test")
    b.add_line("segments", [{"sid": "SEG-A"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("segments", fk_col="seg_id", direction="in", required=True)],
        tracked_properties=["score", "mid_col", "rare_col"],
    )
    b.build()
    pat = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())["patterns"]["cp"]
    # score: fill=1.0 → excluded by MAX_FILL_RATE
    assert "score" not in pat["prop_columns"]
    assert "score" in pat["excluded_properties"]
    # mid_col: fill=0.5 → included
    assert "mid_col" in pat["prop_columns"]
    # rare_col: fill=0.0 → excluded by MIN_FILL_RATE
    assert "rare_col" not in pat["prop_columns"]
    assert "rare_col" in pat["excluded_properties"]


def test_builder_prop_columns_full_coverage_string(tmp_path):
    import json

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [
            {"cid": f"C-{i}", "seg_id": "SEG-A", "nation": "FRANCE"}
            for i in range(10)
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_line("segments", [{"sid": "SEG-A"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cust_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("segments", fk_col="seg_id", direction="in", required=True)],
        tracked_properties=["nation"],
    )
    b.build()

    sphere_json = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_json["patterns"]["cust_pattern"]

    assert "nation" in pat["prop_columns"]
    assert "nation" not in pat["excluded_properties"]


def test_builder_prop_columns_full_coverage_boolean(tmp_path):
    import json

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [
            {"cid": f"C-{i}", "seg_id": "SEG-A", "active": True}
            for i in range(10)
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_line("segments", [{"sid": "SEG-A"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cust_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("segments", fk_col="seg_id", direction="in", required=True)],
        tracked_properties=["active"],
    )
    b.build()

    sphere_json = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_json["patterns"]["cust_pattern"]

    assert "active" not in pat["prop_columns"]
    assert "active" in pat["excluded_properties"]


def test_e2e_property_fill_in_delta(tmp_path):
    """Property null → negative delta in the corresponding dimension."""
    from hypertopos.sphere import HyperSphere

    b = GDSBuilder("s", str(tmp_path / "gds"))
    # 5 customers: C-4 has region=null
    b.add_line(
        "customers",
        [
            {"cid": "C-1", "seg_id": "SEG-A", "name": "A", "region": "EMEA"},
            {"cid": "C-2", "seg_id": "SEG-A", "name": "B", "region": "EMEA"},
            {"cid": "C-3", "seg_id": "SEG-B", "name": "C", "region": "APAC"},
            {"cid": "C-4", "seg_id": "SEG-A", "name": "D", "region": None},  # missing region
            {"cid": "C-5", "seg_id": "SEG-B", "name": None, "region": "APAC"},  # missing name
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_line("segments", [{"sid": "SEG-A"}, {"sid": "SEG-B"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cust_pattern",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("segments", fk_col="seg_id", direction="in", required=True)],
        tracked_properties=["name", "region"],
    )
    path = b.build()

    hs = HyperSphere.open(path)
    sess = hs.session("e2e-test")
    # C-4 has region=null → region fill dim = 0.0 → should be negative delta
    polygon = sess._engine.build_polygon("C-4", "cust_pattern", sess._manifest)
    assert len(polygon.delta) == 3  # 1 edge + 2 props
    # region is dim 2 (index 2): fill=0.0, mu≈0.8 → delta should be negative
    assert polygon.delta[2] < 0.0
    # C-1 has all fields → both fill dims = 1.0 → delta positive (mu < 1.0)
    polygon_c1 = sess._engine.build_polygon("C-1", "cust_pattern", sess._manifest)
    assert polygon_c1.delta[1] > 0.0  # name=present
    assert polygon_c1.delta[2] > 0.0  # region=present
    sess.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# edge_max per relation — continuous shape vector mode
# ---------------------------------------------------------------------------


def test_edge_max_zero_raises(tmp_path):
    """edge_max=0 must raise ValueError, not silently produce NaN."""
    import pytest

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", [{"cid": "C-1", "cnt": 0}], key_col="cid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("customers", fk_col="cnt", direction="in", edge_max=0),
        ],
    )
    with pytest.raises(ValueError, match="edge_max must be >= 1"):
        b.build()


def test_edge_max_non_numeric_column_raises(tmp_path):
    """edge_max on a string column must raise ValueError with clear message."""
    import pytest

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", [{"cid": "C-1", "seg_id": "S-1"}], key_col="cid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("customers", fk_col="seg_id", direction="in", edge_max=5),
        ],
    )
    with pytest.raises(ValueError, match="numeric count column"):
        b.build()


def test_mixed_edge_max_raises(tmp_path):
    """Pattern with some relations having edge_max and some without → ValueError."""
    import pytest

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line("customers", [{"cid": "C-1", "cnt": 2, "seg_id": "S-1"}], key_col="cid", source_id="test")
    b.add_line("segments", [{"sid": "S-1"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("customers", fk_col=None, direction="self"),
            RelationSpec("customers", fk_col="cnt", direction="in", edge_max=5),
            RelationSpec("segments", fk_col="seg_id", direction="in"),  # no edge_max — mixed
        ],
    )
    with pytest.raises(ValueError, match="mixed edge_max"):
        b.build()


def test_continuous_mode_edge_max(tmp_path):
    """edge_max=3: sphere.json has correct list; sphere is navigable."""
    import json
    from pathlib import Path

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [
            {"cid": "C-1", "order_count": 0},
            {"cid": "C-2", "order_count": 1},
            {"cid": "C-3", "order_count": 3},
            {"cid": "C-4", "order_count": 5},  # clamped → shape = 1.0
            {"cid": "C-5", "order_count": 2},
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("customers", fk_col=None, direction="self"),
            RelationSpec(
                "customers", fk_col="order_count", direction="in", required=False, edge_max=3
            ),
        ],
    )
    out = b.build()

    sphere_json = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
    em = sphere_json["patterns"]["cp"]["edge_max"]
    assert em is not None, "edge_max must be a list when any relation has edge_max set"
    assert em[0] == 1, "self-relation must get edge_max=1"
    assert em[1] == 3, "explicit edge_max=3 must be preserved"

    from hypertopos.sphere import HyperSphere

    sess = HyperSphere.open(out).session("test")
    nav = sess.navigator()
    nav.goto("C-4", "customers")
    poly = nav.current_polygon("cp")
    assert poly.delta is not None
    assert len(poly.delta) == 2
    sess.__exit__(None, None, None)


def test_builder_zero_theta_no_anomalies(tmp_path):
    """When all shape vectors are zero, theta_norm=0 and no entity must be marked anomalous.

    If every entity has the same degenerate geometry (zero shape vector), compute_stats
    produces theta=zeros. The builder must write is_anomaly=False for all rows, not True.
    """
    import lance

    b = GDSBuilder("s", str(tmp_path / "gds"))
    # Use a pattern with only a self-relation so the shape vector is constant [1.0]
    # for all entities. compute_stats with constant input → delta_norms all=0 → theta=zeros.
    b.add_line("items", [{"iid": f"I-{i}"} for i in range(10)], key_col="iid", source_id="test")
    b.add_pattern(
        "item_pattern",
        pattern_type="anchor",
        entity_line="items",
        relations=[
            RelationSpec("items", fk_col=None, direction="self"),
        ],
    )
    b.build()

    geo_path = str(tmp_path / "gds" / "geometry" / "item_pattern" / "v=1" / "data.lance")
    geo = lance.dataset(geo_path).to_table()

    anomaly_flags = geo["is_anomaly"].to_pylist()
    assert not any(anomaly_flags), (
        f"Expected all is_anomaly=False for degenerate pattern (theta_norm=0), "
        f"got {sum(anomaly_flags)} anomalies out of {len(anomaly_flags)}"
    )


def test_delta_rank_pct_minimum_norm_gets_zero_percentile():
    """Entity with the minimum delta_norm must get delta_rank_pct=0, not 100.

    When ALL entities have delta_norm=0.0, each entity is at the minimum.
    Expected: pct=0 (0% of entities have lower norm than 0.0).
    Regression test for BUG-4: searchsorted side='right' returned n (past-end)
    for all-zero norms, producing pct=100.0 instead of 0.0.
    """
    delta_norms = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    n = len(delta_norms)
    sorted_norms = np.sort(delta_norms)
    ranks = np.searchsorted(sorted_norms, delta_norms, side="left")
    pcts = (ranks / n * 100).astype(np.float32)
    assert (pcts == 0.0).all(), f"Expected all 0.0 pct for all-zero norms, got {pcts}"


def test_delta_rank_pct_distinct_norms_ranked_correctly():
    """Distinct delta_norms must produce strictly increasing percentile ranks.

    Minimum entity gets pct=0, each next entity gets a strictly higher percentile.
    """
    delta_norms = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    n = len(delta_norms)
    sorted_norms = np.sort(delta_norms)
    ranks = np.searchsorted(sorted_norms, delta_norms, side="left")
    pcts = (ranks / n * 100).astype(np.float32)
    # Minimum (0.1): 0 elements strictly below it → pct=0
    assert pcts[0] == 0.0, f"Expected pct=0 for minimum norm, got {pcts[0]}"
    # Each next is strictly higher
    assert all(pcts[i] < pcts[i + 1] for i in range(len(pcts) - 1)), (
        f"Expected strictly increasing pcts, got {pcts}"
    )


def test_delta_rank_pct_consistent_after_two_builds(tmp_path):
    """Entities built in a second pass must get globally-consistent delta_rank_pct.

    Without the fix, entities added in a second append batch get their percentile
    computed only against that small batch — so the entity with the highest delta_norm
    in the batch receives delta_rank_pct=100.0 even if it is mid-population globally.
    With the fix, recompute_delta_rank_pct re-ranks all entities against the full population.

    Setup:
    - First batch: 10 GL entries, 8 have a customer FK (low delta), 2 do not (high delta).
    - Second batch (append): 1 synthetic row with a mid-range delta_norm (0.3) explicitly
      written with delta_rank_pct=100.0 to simulate the per-batch bug.
    - After recompute: C-NEW must have delta_rank_pct well below 100.0.
    """
    from datetime import UTC, datetime

    import lance
    from hypertopos.builder.builder import GEOMETRY_EVENT_SCHEMA
    from hypertopos.storage.writer import GDSWriter

    # Build a sphere where entities have varied delta_norms:
    # 8 entries have customer FK (shape≈[1,1] → low delta), 2 have no FK (shape≈[1,0] → higher).
    b1 = GDSBuilder("s", str(tmp_path / "gds"))
    b1.add_line("customers", [{"cid": f"C-{i:03d}"} for i in range(10)], key_col="cid", source_id="test")
    b1.add_line(
        "gl_entries",
        [{"doc": f"D-{i:03d}", "cust_id": f"C-{i:03d}"} for i in range(8)]
        + [
            {"doc": "D-008", "cust_id": None},
            {"doc": "D-009", "cust_id": None},
        ],
        key_col="doc",
        source_id="test",
        role="event",
    )
    b1.add_pattern(
        "gp",
        pattern_type="event",
        entity_line="gl_entries",
        relations=[
            RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
        ],
        anomaly_percentile=80.0,
    )
    out = b1.build()

    writer = GDSWriter(out)
    geo_path = tmp_path / "gds" / "geometry" / "gp" / "v=1" / "data.lance"
    existing_tbl = lance.dataset(str(geo_path)).to_table(columns=["delta_norm"])
    all_norms_before = [v for v in existing_tbl["delta_norm"].to_pylist() if v is not None]
    max_norm = max(all_norms_before)
    min_norm = min(all_norms_before)
    # mid_norm is strictly between min and max so C-NEW is not the global maximum
    mid_norm = float(min_norm) + (float(max_norm) - float(min_norm)) * 0.3
    assert mid_norm < max_norm, "mid_norm must be below global max for the test to be meaningful"

    # Append one synthetic geometry row with the pre-fix bad pct=100.0
    now = datetime.now(UTC)
    new_row = pa.table(
        {
            "primary_key": pa.array(["C-NEW"], type=pa.string()),
            "scale": pa.array([1], type=pa.int32()),
            "delta": pa.array([[mid_norm]], type=pa.list_(pa.float32())),
            "delta_norm": pa.array([mid_norm], type=pa.float32()),
            "delta_rank_pct": pa.array([100.0], type=pa.float32()),  # bad pre-fix value
            "is_anomaly": pa.array([False], type=pa.bool_()),
            "conformal_p": pa.array([0.5], type=pa.float32()),
            "n_anomalous_dims": pa.array([0], type=pa.int32()),
            "entity_keys": pa.array([[]], type=pa.list_(pa.string())),
            "last_refresh_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
            "updated_at": pa.array([now], type=pa.timestamp("us", tz="UTC")),
        },
        schema=GEOMETRY_EVENT_SCHEMA,
    )
    # append_geometry automatically recomputes delta_rank_pct globally
    writer.append_geometry(new_row, "gp", version=1)

    # After fix: verify global consistency
    tbl_fixed = lance.dataset(str(geo_path)).to_table(
        columns=["primary_key", "delta_rank_pct", "delta_norm"]
    )
    pct_fixed = {
        tbl_fixed["primary_key"][i].as_py(): tbl_fixed["delta_rank_pct"][i].as_py()
        for i in range(tbl_fixed.num_rows)
    }
    norm_fixed = {
        tbl_fixed["primary_key"][i].as_py(): tbl_fixed["delta_norm"][i].as_py()
        for i in range(tbl_fixed.num_rows)
    }
    # The entity with the global maximum norm must have the highest delta_rank_pct.
    # With side="left" searchsorted the maximum entity gets rank=(n-1), so pct=(n-1)/n*100 < 100.
    max_norm_key = max(norm_fixed, key=norm_fixed.__getitem__)
    assert pct_fixed[max_norm_key] == max(pct_fixed.values()), (
        f"Entity with global max delta_norm ({max_norm_key}) must have the highest delta_rank_pct"
    )
    # C-NEW (mid-range norm) must NOT be ranked at the top after global recompute
    assert pct_fixed["C-NEW"] < 99.0, (
        f"After fix: C-NEW (mid-range norm) must have delta_rank_pct < 99.0, "
        f"got {pct_fixed['C-NEW']:.1f}"
    )


def test_continuous_mode_shape_values(tmp_path):
    """C-4 (count=5, clamped) and C-3 (count=3) must have same shape[1]; C-1 shape[1]=0."""
    import json
    from pathlib import Path

    import lance

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [
            {"cid": "C-1", "cnt": 0},
            {"cid": "C-2", "cnt": 3},
            {"cid": "C-3", "cnt": 3},
            {"cid": "C-4", "cnt": 5},  # clamped to 3
            {"cid": "C-5", "cnt": 1},
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[
            RelationSpec("customers", fk_col=None, direction="self"),
            RelationSpec("customers", fk_col="cnt", direction="in", required=False, edge_max=3),
        ],
    )
    out = b.build()

    sphere_json = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_json["patterns"]["cp"]
    mu = np.array(pat["mu"], dtype=np.float32)
    sigma = np.maximum(np.array(pat["sigma_diag"], dtype=np.float32), 1e-2)

    geo = lance.dataset(str(tmp_path / "gds" / "geometry" / "cp" / "v=1" / "data.lance")).to_table()
    rows = {
        geo["primary_key"][i].as_py(): np.array(geo["delta"][i].as_py(), dtype=np.float32)
        for i in range(geo.num_rows)
    }

    shape = {k: rows[k] * sigma + mu for k in rows}

    assert abs(shape["C-3"][1] - shape["C-4"][1]) < 1e-3, (
        f"C-3 shape[1]={shape['C-3'][1]:.4f} != C-4 shape[1]={shape['C-4'][1]:.4f}"
    )
    assert abs(shape["C-1"][1] - 0.0) < 1e-3, f"C-1 shape[1]={shape['C-1'][1]:.4f} != 0.0"


# ---------------------------------------------------------------------------
# GDSBuilder.build() must persist geometry_stats cache
# ---------------------------------------------------------------------------


def test_builder_writes_geometry_stats(tmp_path):
    """GDSBuilder.build() must create geometry_stats JSON for each pattern."""
    import json
    from pathlib import Path

    b = _make_minimal_builder(tmp_path)
    out = b.build()

    stats_path = Path(out) / "_gds_meta" / "geometry_stats" / "gl_entry_pattern_v1.json"
    assert stats_path.exists(), f"geometry_stats cache file not found at {stats_path}"

    stats = json.loads(stats_path.read_text())
    assert stats["pattern_id"] == "gl_entry_pattern"
    assert stats["version"] == 1
    assert stats["total_entities"] == 8  # 8 gl_entries
    assert isinstance(stats["theta_norm"], float)
    assert "percentiles" in stats
    assert "p50" in stats["percentiles"]
    assert "p95" in stats["percentiles"]
    assert "max" in stats["percentiles"]


def test_builder_geometry_stats_theta_norm_matches_pattern(tmp_path):
    """theta_norm in geometry_stats must equal np.linalg.norm(theta) from sphere.json."""
    import json
    from pathlib import Path

    b = _make_minimal_builder(tmp_path)
    out = b.build()

    sphere_data = json.loads((Path(out) / "_gds_meta" / "sphere.json").read_text())
    theta = np.array(sphere_data["patterns"]["gl_entry_pattern"]["theta"], dtype=np.float64)
    expected_theta_norm = float(np.linalg.norm(theta))

    stats = json.loads(
        (Path(out) / "_gds_meta" / "geometry_stats" / "gl_entry_pattern_v1.json").read_text()
    )
    assert abs(stats["theta_norm"] - expected_theta_norm) < 1e-4, (
        f"theta_norm mismatch: stats={stats['theta_norm']}, expected={expected_theta_norm}"
    )


def test_builder_geometry_stats_anomaly_count_consistent(tmp_path):
    """total_anomalies in geometry_stats must match is_anomaly count in geometry table."""
    import json
    from pathlib import Path

    import lance

    b = _make_minimal_builder(tmp_path)
    out = b.build()

    stats = json.loads(
        (Path(out) / "_gds_meta" / "geometry_stats" / "gl_entry_pattern_v1.json").read_text()
    )

    geo_path = str(Path(out) / "geometry" / "gl_entry_pattern" / "v=1" / "data.lance")
    geo = lance.dataset(geo_path).to_table(columns=["is_anomaly"])
    actual_anomaly_count = sum(1 for v in geo["is_anomaly"].to_pylist() if v)

    assert stats["total_anomalies"] == actual_anomaly_count, (
        f"total_anomalies mismatch: stats={stats['total_anomalies']}, "
        f"geometry={actual_anomaly_count}"
    )


# ---------------------------------------------------------------------------
# SIGMA_EPS_PROP floor for prop_column dimensions
# ---------------------------------------------------------------------------


def test_prop_sigma_floor_caps_deltas(tmp_path):
    """Prop_column dims get SIGMA_EPS_PROP=0.2, capping deltas at ~5.

    When 99/100 entities have the property (fill=1.0) and 1 doesn't (fill=0.0),
    natural sigma ≈ 0.0995, well below SIGMA_EPS_PROP=0.2. Without the floor,
    SIGMA_EPS=0.01 produces deltas up to ~100; with the floor, deltas stay ≤ 5.
    """
    import json
    from pathlib import Path

    import lance

    b = GDSBuilder("test_sphere", str(tmp_path / "gds"))

    # 100 entities: 99 have the property, 1 doesn't → sigma ≈ 0.0995
    # Without SIGMA_EPS_PROP floor: sigma clamped to 0.01 → delta up to ~100
    # With SIGMA_EPS_PROP=0.2: delta capped at ~5
    n = 100
    data = pa.table(
        {
            "primary_key": [f"E-{i:03d}" for i in range(n)],
            "fk_other": [f"O-{i % 10:03d}" for i in range(n)],
            "tracked_prop": [True] * 99 + [None] * 1,
        }
    )

    b.add_line("entities", data, key_col="primary_key", source_id="test")
    b.add_line(
        "others",
        pa.table({"primary_key": [f"O-{i:03d}" for i in range(10)]}),
        key_col="primary_key",
        source_id="test",
    )
    b.add_pattern(
        "entity_pattern",
        "anchor",
        "entities",
        relations=[RelationSpec("others", "fk_other", direction="in")],
        tracked_properties=["tracked_prop"],
    )

    path = b.build()

    # Read sphere.json and check sigma_diag
    sphere_data = json.loads((Path(path) / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_data["patterns"]["entity_pattern"]
    sigma = pat["sigma_diag"]

    # dim 0 = relation (fk_other) — all entities have it, std=0 → SIGMA_EPS=0.01
    # dim 1 = prop_column (tracked_prop) — must be >= 0.2 (SIGMA_EPS_PROP)
    assert sigma[1] >= 0.2, f"Prop sigma {sigma[1]} should be >= 0.2"

    # Read geometry to check delta values
    geo_path = Path(path) / "geometry" / "entity_pattern" / "v=1" / "data.lance"
    geo = lance.dataset(str(geo_path)).to_table()
    deltas = [row.as_py() for row in geo["delta"]]

    # Max prop_column delta should be around ±5, not ±100
    prop_deltas = [d[1] for d in deltas]
    max_abs = max(abs(d) for d in prop_deltas)
    assert max_abs <= 6.0, (
        f"Prop deltas exceed ±6: max={max_abs:.2f} — SIGMA_EPS_PROP floor not applied"
    )


def test_no_prop_columns_sigma_unchanged(tmp_path):
    """Pattern without prop_columns uses standard SIGMA_EPS=0.01."""
    import json
    from pathlib import Path

    b = GDSBuilder("test_sphere", str(tmp_path / "gds"))

    # All entities identical FK → sigma=0 → clamped to SIGMA_EPS=0.01
    n = 20
    data = pa.table(
        {
            "primary_key": [f"E-{i:03d}" for i in range(n)],
            "fk_other": [f"O-{i % 5:03d}" for i in range(n)],
        }
    )

    b.add_line("entities", data, key_col="primary_key", source_id="test")
    b.add_line(
        "others",
        pa.table({"primary_key": [f"O-{i:03d}" for i in range(5)]}),
        key_col="primary_key",
        source_id="test",
    )
    b.add_pattern(
        "entity_pattern",
        "anchor",
        "entities",
        relations=[RelationSpec("others", "fk_other", direction="in")],
        # No tracked_properties — SIGMA_EPS_PROP must NOT apply
    )

    path = b.build()

    sphere_data = json.loads((Path(path) / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_data["patterns"]["entity_pattern"]
    sigma = pat["sigma_diag"]

    assert len(sigma) == 1, f"Expected 1 dim (no props), got {len(sigma)}"
    # All entities have a valid FK, so std=0, clamped to SIGMA_EPS=0.01
    # It must NOT be 0.2 (SIGMA_EPS_PROP) since there are no prop_columns
    assert sigma[0] < 0.2, (
        f"sigma[0]={sigma[0]:.4f} — SIGMA_EPS_PROP incorrectly applied "
        "to a pattern without prop_columns"
    )


# ---------------------------------------------------------------------------
# Streaming three-pass stats
# ---------------------------------------------------------------------------


def _make_streaming_builder(tmp_path, n_entities=500, suffix=""):
    """Build a sphere with enough entities to trigger chunked/streaming path."""
    rng = np.random.default_rng(42)
    b = GDSBuilder("streaming_test", str(tmp_path / f"gds_stream{suffix}"))
    customers = [
        {"cust_id": f"C-{i:04d}", "cc_id": f"CC-{rng.integers(1, 4)}"} for i in range(n_entities)
    ]
    b.add_line("customers", customers, key_col="cust_id", source_id="test")
    b.add_line(
        "company_codes",
        [
            {"cc_id": "CC-1"},
            {"cc_id": "CC-2"},
            {"cc_id": "CC-3"},
        ],
        key_col="cc_id",
        source_id="test",
    )
    gl_entries = [
        {
            "doc": f"D-{i:04d}",
            "cust_id": f"C-{rng.integers(0, n_entities):04d}",
            "cc_id": f"CC-{rng.integers(1, 4)}",
        }
        for i in range(n_entities)
    ]
    # Add some null FKs for diversity
    for i in range(0, n_entities, 7):
        gl_entries[i]["cust_id"] = None
    for i in range(0, n_entities, 11):
        gl_entries[i]["cc_id"] = None
    b.add_line("gl_entries", gl_entries, key_col="doc", source_id="test", role="event")
    b.add_pattern(
        "gl_pattern",
        pattern_type="event",
        entity_line="gl_entries",
        relations=[
            RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
            RelationSpec("company_codes", fk_col="cc_id", direction="in", required=False),
        ],
        anomaly_percentile=90.0,
    )
    return b


def test_streaming_stats_match_full_population(tmp_path, monkeypatch):
    """Streaming path must produce identical mu/sigma/theta/norms as non-streaming."""
    import json
    from pathlib import Path

    import hypertopos.builder.builder as builder_mod
    import lance

    # Path A: streaming (chunk_size=100, so 500 entities triggers streaming)
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 100)
    b_stream = _make_streaming_builder(tmp_path, n_entities=500, suffix="_a")
    out_a = b_stream.build()
    sphere_a = json.loads(
        (Path(out_a) / "_gds_meta" / "sphere.json").read_text(),
    )
    geo_a = lance.dataset(
        str(Path(out_a) / "geometry" / "gl_pattern" / "v=1" / "data.lance"),
    ).to_table()

    # Path B: non-streaming (chunk_size huge, all in-memory)
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 1_000_000)
    b_full = _make_streaming_builder(tmp_path, n_entities=500, suffix="_b")
    out_b = b_full.build()
    sphere_b = json.loads(
        (Path(out_b) / "_gds_meta" / "sphere.json").read_text(),
    )
    geo_b = lance.dataset(
        str(Path(out_b) / "geometry" / "gl_pattern" / "v=1" / "data.lance"),
    ).to_table()

    pat_a = sphere_a["patterns"]["gl_pattern"]
    pat_b = sphere_b["patterns"]["gl_pattern"]

    # mu, sigma, theta must be equal within float32 tolerance
    mu_a = np.array(pat_a["mu"], dtype=np.float32)
    mu_b = np.array(pat_b["mu"], dtype=np.float32)
    np.testing.assert_allclose(mu_a, mu_b, atol=1e-5, rtol=1e-5, err_msg="mu mismatch")

    sigma_a = np.array(pat_a["sigma_diag"], dtype=np.float32)
    sigma_b = np.array(pat_b["sigma_diag"], dtype=np.float32)
    np.testing.assert_allclose(sigma_a, sigma_b, atol=1e-5, rtol=1e-5, err_msg="sigma mismatch")

    theta_a = np.array(pat_a["theta"], dtype=np.float32)
    theta_b = np.array(pat_b["theta"], dtype=np.float32)
    np.testing.assert_allclose(theta_a, theta_b, atol=1e-4, rtol=1e-4, err_msg="theta mismatch")

    # Compare geometry rows: sort by primary_key for stable comparison
    def _sorted_geo(geo):
        keys = geo["primary_key"].to_pylist()
        norms = geo["delta_norm"].to_pylist()
        ranks = geo["delta_rank_pct"].to_pylist()
        anomaly = geo["is_anomaly"].to_pylist()
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        return (
            [keys[i] for i in order],
            np.array([norms[i] for i in order], dtype=np.float32),
            np.array([ranks[i] for i in order], dtype=np.float32),
            [anomaly[i] for i in order],
        )

    keys_a, norms_a, ranks_a, anom_a = _sorted_geo(geo_a)
    keys_b, norms_b, ranks_b, anom_b = _sorted_geo(geo_b)

    assert keys_a == keys_b, "Primary keys differ"
    np.testing.assert_allclose(
        norms_a, norms_b, atol=1e-4, rtol=1e-4, err_msg="delta_norm mismatch"
    )
    np.testing.assert_allclose(
        ranks_a, ranks_b, atol=1e-2, rtol=1e-2, err_msg="delta_rank_pct mismatch"
    )
    assert anom_a == anom_b, "is_anomaly flags differ"


def test_streaming_stats_with_dim_weights(tmp_path, monkeypatch):
    """Streaming path with dimension_weights='auto' matches non-streaming."""
    import json
    from pathlib import Path

    import hypertopos.builder.builder as builder_mod
    import lance

    n_entities = 500

    def _make_weighted_builder(sfx):
        # Fresh rng per call to ensure identical data
        r = np.random.default_rng(42)
        b = GDSBuilder("wt", str(tmp_path / f"gds_wt{sfx}"))
        customers = [
            {"cust_id": f"C-{i:04d}", "seg_id": f"S-{r.integers(1, 3)}"} for i in range(n_entities)
        ]
        b.add_line("customers", customers, key_col="cust_id", source_id="test")
        b.add_line("segments", [{"sid": "S-1"}, {"sid": "S-2"}], key_col="sid", source_id="test")
        gl = [
            {
                "doc": f"D-{i:04d}",
                "cust_id": f"C-{r.integers(0, n_entities):04d}",
                "seg_id": f"S-{r.integers(1, 3)}",
            }
            for i in range(n_entities)
        ]
        for i in range(0, n_entities, 5):
            gl[i]["cust_id"] = None
        b.add_line("gl", gl, key_col="doc", source_id="test", role="event")
        b.add_pattern(
            "glp",
            pattern_type="event",
            entity_line="gl",
            relations=[
                RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
                RelationSpec("segments", fk_col="seg_id", direction="in", required=False),
            ],
            dimension_weights="auto",
            anomaly_percentile=90.0,
        )
        return b

    # Path A: streaming
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 100)
    out_a = _make_weighted_builder("_a").build()
    sphere_a = json.loads(
        (Path(out_a) / "_gds_meta" / "sphere.json").read_text(),
    )

    # Path B: non-streaming
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 1_000_000)
    out_b = _make_weighted_builder("_b").build()
    sphere_b = json.loads(
        (Path(out_b) / "_gds_meta" / "sphere.json").read_text(),
    )

    pat_a = sphere_a["patterns"]["glp"]
    pat_b = sphere_b["patterns"]["glp"]

    # dimension_weights must be present and match
    assert "dimension_weights" in pat_a
    assert "dimension_weights" in pat_b
    w_a = np.array(pat_a["dimension_weights"], dtype=np.float32)
    w_b = np.array(pat_b["dimension_weights"], dtype=np.float32)
    np.testing.assert_allclose(w_a, w_b, atol=1e-4, rtol=1e-4, err_msg="dimension_weights mismatch")

    # theta must match
    theta_a = np.array(pat_a["theta"], dtype=np.float32)
    theta_b = np.array(pat_b["theta"], dtype=np.float32)
    np.testing.assert_allclose(
        theta_a, theta_b, atol=1e-4, rtol=1e-4, err_msg="theta mismatch with dim_weights"
    )

    # Geometry norms must match
    geo_a = lance.dataset(
        str(Path(out_a) / "geometry" / "glp" / "v=1" / "data.lance"),
    ).to_table()
    geo_b = lance.dataset(
        str(Path(out_b) / "geometry" / "glp" / "v=1" / "data.lance"),
    ).to_table()

    def _sorted_norms(geo):
        keys = geo["primary_key"].to_pylist()
        norms = geo["delta_norm"].to_pylist()
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        return np.array([norms[i] for i in order], dtype=np.float32)

    np.testing.assert_allclose(
        _sorted_norms(geo_a),
        _sorted_norms(geo_b),
        atol=1e-4,
        rtol=1e-4,
        err_msg="delta_norm mismatch with dim_weights",
    )


def test_dimension_weights_kurtosis_alias(tmp_path, monkeypatch):
    """dimension_weights='kurtosis' must produce same result as 'auto'."""
    import json
    from pathlib import Path

    import hypertopos.builder.builder as builder_mod

    n_entities = 500

    def _make(sfx, dw_value):
        r = np.random.default_rng(42)
        b = GDSBuilder("wk", str(tmp_path / f"gds_wk{sfx}"))
        customers = [
            {"cust_id": f"C-{i:04d}", "seg_id": f"S-{r.integers(1, 3)}"} for i in range(n_entities)
        ]
        b.add_line("customers", customers, key_col="cust_id", source_id="test")
        b.add_line("segments", [{"sid": "S-1"}, {"sid": "S-2"}], key_col="sid", source_id="test")
        gl = [
            {
                "doc": f"D-{i:04d}",
                "cust_id": f"C-{r.integers(0, n_entities):04d}",
                "seg_id": f"S-{r.integers(1, 3)}",
            }
            for i in range(n_entities)
        ]
        b.add_line("gl", gl, key_col="doc", source_id="test", role="event")
        b.add_pattern(
            "glp",
            pattern_type="event",
            entity_line="gl",
            relations=[
                RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
                RelationSpec("segments", fk_col="seg_id", direction="in", required=False),
            ],
            dimension_weights=dw_value,
            anomaly_percentile=90.0,
        )
        return b

    # Force non-streaming path
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 1_000_000)

    out_auto = _make("_auto", "auto").build()
    out_kurt = _make("_kurt", "kurtosis").build()

    sphere_auto = json.loads((Path(out_auto) / "_gds_meta" / "sphere.json").read_text())
    sphere_kurt = json.loads((Path(out_kurt) / "_gds_meta" / "sphere.json").read_text())

    w_auto = sphere_auto["patterns"]["glp"].get("dimension_weights")
    w_kurt = sphere_kurt["patterns"]["glp"].get("dimension_weights")

    assert w_kurt is not None, "kurtosis must produce dimension_weights"
    np.testing.assert_array_equal(w_auto, w_kurt)

    # Also verify streaming path
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 100)

    out_kurt_s = _make("_kurt_s", "kurtosis").build()
    sphere_kurt_s = json.loads((Path(out_kurt_s) / "_gds_meta" / "sphere.json").read_text())
    w_kurt_s = sphere_kurt_s["patterns"]["glp"].get("dimension_weights")

    assert w_kurt_s is not None, "kurtosis must produce dimension_weights in streaming path"
    np.testing.assert_allclose(
        w_kurt, w_kurt_s, atol=1e-4, rtol=1e-4,
        err_msg="kurtosis weights: streaming vs non-streaming mismatch",
    )


def test_invalid_dimension_weights_raises(tmp_path):
    """Unknown dimension_weights string must raise ValueError."""
    import pytest

    b = GDSBuilder("inv", str(tmp_path / "sphere"))
    b.add_line("items", [{"pk": "A"}], key_col="pk", source_id="t")
    with pytest.raises(ValueError, match="dimension_weights='foobar'"):
        b.add_pattern("p", "event", "items", relations=[], dimension_weights="foobar")


def test_streaming_with_tracked_properties(tmp_path, monkeypatch):
    """Streaming path with tracked_properties matches non-streaming."""
    import json
    from pathlib import Path

    import hypertopos.builder.builder as builder_mod
    import lance

    n_entities = 500

    def _make_prop_builder(sfx):
        # Fresh rng per call to ensure identical data
        r = np.random.default_rng(42)
        b = GDSBuilder("tp", str(tmp_path / f"gds_tp{sfx}"))
        customers = []
        for i in range(n_entities):
            c = {
                "cust_id": f"C-{i:04d}",
                "seg_id": f"S-{r.integers(1, 3)}",
                "name": f"Name-{i}" if r.random() > 0.1 else None,
                "region": f"R-{r.integers(1, 4)}" if r.random() > 0.2 else None,
            }
            customers.append(c)
        b.add_line("customers", customers, key_col="cust_id", source_id="test")
        b.add_line("segments", [{"sid": "S-1"}, {"sid": "S-2"}], key_col="sid", source_id="test")
        b.add_pattern(
            "cp",
            pattern_type="anchor",
            entity_line="customers",
            relations=[
                RelationSpec("segments", fk_col="seg_id", direction="in", required=True),
            ],
            tracked_properties=["name", "region"],
            anomaly_percentile=90.0,
        )
        return b

    # Path A: streaming
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 100)
    out_a = _make_prop_builder("_a").build()
    sphere_a = json.loads(
        (Path(out_a) / "_gds_meta" / "sphere.json").read_text(),
    )

    # Path B: non-streaming
    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 1_000_000)
    out_b = _make_prop_builder("_b").build()
    sphere_b = json.loads(
        (Path(out_b) / "_gds_meta" / "sphere.json").read_text(),
    )

    pat_a = sphere_a["patterns"]["cp"]
    pat_b = sphere_b["patterns"]["cp"]

    # prop_columns must match
    assert pat_a["prop_columns"] == pat_b["prop_columns"]
    assert pat_a["excluded_properties"] == pat_b["excluded_properties"]

    # mu/sigma must match (includes prop dims)
    mu_a = np.array(pat_a["mu"], dtype=np.float32)
    mu_b = np.array(pat_b["mu"], dtype=np.float32)
    np.testing.assert_allclose(mu_a, mu_b, atol=1e-5, rtol=1e-5, err_msg="mu mismatch with props")

    sigma_a = np.array(pat_a["sigma_diag"], dtype=np.float32)
    sigma_b = np.array(pat_b["sigma_diag"], dtype=np.float32)
    np.testing.assert_allclose(
        sigma_a, sigma_b, atol=1e-5, rtol=1e-5, err_msg="sigma mismatch with props"
    )

    # Geometry comparison
    geo_a = lance.dataset(
        str(Path(out_a) / "geometry" / "cp" / "v=1" / "data.lance"),
    ).to_table()
    geo_b = lance.dataset(
        str(Path(out_b) / "geometry" / "cp" / "v=1" / "data.lance"),
    ).to_table()

    def _sorted_norms(geo):
        keys = geo["primary_key"].to_pylist()
        norms = geo["delta_norm"].to_pylist()
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        return np.array([norms[i] for i in order], dtype=np.float32)

    np.testing.assert_allclose(
        _sorted_norms(geo_a),
        _sorted_norms(geo_b),
        atol=1e-4,
        rtol=1e-4,
        err_msg="delta_norm mismatch with tracked_properties",
    )


def test_streaming_complex_mode_falls_back_to_chunked(tmp_path, monkeypatch):
    """Complex modes (group_by, GMM, Mahalanobis) use chunked, not streaming."""
    from unittest.mock import patch

    import hypertopos.builder.builder as builder_mod

    monkeypatch.setattr(builder_mod, "GEOMETRY_CHUNK_SIZE", 100)

    rng = np.random.default_rng(42)
    b = GDSBuilder("fb", str(tmp_path / "gds_fb"))
    gl = [
        {
            "doc": f"D-{i:04d}",
            "cust_id": f"C-{rng.integers(0, 50):04d}",
            "region": f"R-{rng.integers(1, 3)}",
        }
        for i in range(500)
    ]
    b.add_line("gl", gl, key_col="doc", source_id="test", role="event")
    b.add_line("customers", [{"cust_id": f"C-{i:04d}"} for i in range(50)], key_col="cust_id", source_id="test")
    b.add_pattern(
        "gp",
        pattern_type="event",
        entity_line="gl",
        relations=[
            RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
        ],
        group_by_property="region",
        anomaly_percentile=90.0,
    )

    with patch.object(
        builder_mod.GDSBuilder,
        "_build_and_write_chunked",
        wraps=b._build_and_write_chunked,
    ) as mock_chunked:
        b.build()
        assert mock_chunked.call_count == 1, (
            "Complex mode (group_by) must use _build_and_write_chunked"
        )


def test_welford_batch_update_accuracy():
    """Welford batch update must produce correct mean and variance."""
    from hypertopos.builder._stats import welford_batch_update

    rng = np.random.default_rng(123)
    data = rng.normal(5.0, 2.0, size=(1000, 3)).astype(np.float32)

    # One-shot
    mu_exact = data.mean(axis=0).astype(np.float64)
    var_exact = data.var(axis=0, ddof=0).astype(np.float64)

    # Streamed in chunks of 100
    running_mean = np.zeros(3, dtype=np.float64)
    running_m2 = np.zeros(3, dtype=np.float64)
    n_total = 0
    for i in range(0, 1000, 100):
        batch = data[i : i + 100]
        running_mean, running_m2, n_total = welford_batch_update(
            running_mean,
            running_m2,
            n_total,
            batch,
        )

    np.testing.assert_allclose(running_mean, mu_exact, atol=1e-5)
    var_streaming = running_m2 / n_total
    np.testing.assert_allclose(var_streaming, var_exact, atol=1e-4)


def test_source_id_persisted_in_sphere_json(tmp_path):
    """Lines built with source_id must have it in sphere.json."""
    import json

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "accounts",
        [{"acc_id": "A-1", "balance": 100}],
        key_col="acc_id",
        source_id="accounts",
    )
    b.add_line(
        "accounts_stress",
        [{"acc_id": "A-1", "stress_flag": True}],
        key_col="acc_id",
        source_id="accounts",
    )
    b.add_line(
        "customers",
        [{"cust_id": "C-1", "name": "Alpha"}],
        key_col="cust_id",
        source_id="customers",
    )
    b.add_pattern(
        "acc_pat",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[RelationSpec("accounts", fk_col=None, direction="self")],
    )
    b.add_pattern(
        "stress_pat",
        pattern_type="anchor",
        entity_line="accounts_stress",
        relations=[RelationSpec("accounts_stress", fk_col=None, direction="self")],
    )
    b.add_pattern(
        "cust_pat",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("customers", fk_col=None, direction="self")],
    )
    b.build()

    sphere_json = json.loads(
        (tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text()
    )
    lines = sphere_json["lines"]
    assert lines["accounts"]["source_id"] == "accounts"
    assert lines["accounts_stress"]["source_id"] == "accounts"
    assert lines["customers"]["source_id"] == "customers"


def test_builder_computes_dim_percentiles(tmp_path):
    """Builder writes dim_percentiles for numeric float columns on entity lines."""
    import json

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [
            {"cid": "C-1", "seg_id": "SEG-A", "avg_late_days": 10.0, "max_late_days": 50.0},
            {"cid": "C-2", "seg_id": "SEG-A", "avg_late_days": 20.0, "max_late_days": 80.0},
            {"cid": "C-3", "seg_id": "SEG-B", "avg_late_days": 25.0, "max_late_days": 100.0},
            {"cid": "C-4", "seg_id": "SEG-A", "avg_late_days": 30.0, "max_late_days": 120.0},
            {"cid": "C-5", "seg_id": "SEG-B", "avg_late_days": 322.0, "max_late_days": 400.0},
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_line("segments", [{"sid": "SEG-A"}, {"sid": "SEG-B"}], key_col="sid", source_id="test")
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("segments", fk_col="seg_id", direction="in", required=True)],
    )
    b.build()

    sphere_json = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_json["patterns"]["cp"]
    assert "dim_percentiles" in pat
    dp = pat["dim_percentiles"]
    assert "avg_late_days" in dp
    assert "max_late_days" in dp
    assert dp["avg_late_days"]["min"] == 10.0
    assert dp["avg_late_days"]["max"] == 322.0
    assert dp["avg_late_days"]["p50"] > 0
    assert set(dp["avg_late_days"].keys()) == {"min", "p25", "p50", "p75", "p99", "max"}


def test_builder_dim_percentiles_skips_string_columns(tmp_path):
    """dim_percentiles must skip string columns and primary_key."""
    import json

    b = GDSBuilder("s", str(tmp_path / "gds"))
    b.add_line(
        "customers",
        [
            {"cid": "C-1", "name": "Alpha", "score": 1.0},
            {"cid": "C-2", "name": "Beta", "score": 2.0},
            {"cid": "C-3", "name": "Gamma", "score": 3.0},
        ],
        key_col="cid",
        source_id="test",
    )
    b.add_pattern(
        "cp",
        pattern_type="anchor",
        entity_line="customers",
        relations=[RelationSpec("customers", fk_col=None, direction="self")],
    )
    b.build()

    sphere_json = json.loads((tmp_path / "gds" / "_gds_meta" / "sphere.json").read_text())
    pat = sphere_json["patterns"]["cp"]
    dp = pat.get("dim_percentiles", {})
    assert "cid" not in dp
    assert "name" not in dp
    if dp:
        assert "score" in dp
