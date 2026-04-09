# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).parent / "gds" / "sales_sphere"


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _write_lance(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(path), mode="overwrite")


def generate_sphere_json() -> None:
    sphere = {
        "sphere_id": "sales_sphere",
        "name": "Sales Sphere",
        "lines": {
            "customers": {
                "line_id": "customers",
                "entity_type": "customer",
                "line_role": "anchor",
                "pattern_id": "customer_pattern",
                "partitioning": {"mode": "static", "columns": ["region"]},
                "versions": [1],
                "columns": [
                    {"name": "primary_key", "type": "string"},
                    {"name": "name", "type": "string"},
                    {"name": "region", "type": "string"},
                    {"name": "balance", "type": "float64"},
                ],
            },
            "products": {
                "line_id": "products",
                "entity_type": "product",
                "line_role": "anchor",
                "pattern_id": "product_pattern",
                "partitioning": {"mode": "static", "columns": []},
                "versions": [1],
            },
        },
        "patterns": {
            "customer_pattern": {
                "pattern_id": "customer_pattern",
                "entity_type": "customer",
                "pattern_type": "anchor",
                "version": 1,
                "status": "production",
                "relations": [
                    {"line_id": "products", "direction": "out", "required": True},
                    {"line_id": "stores", "direction": "in", "required": False},
                ],
                "mu": [0.7, 0.4],
                "sigma_diag": [0.2, 0.3],
                "theta": [7.5, 5.0],
                "population_size": 100,  # z-scored: [1.5/0.2, 1.5/0.3]
                "computed_at": "2024-01-01T00:00:00+00:00",
            }
        },
        "aliases": {},
        "storage": {
            "geometry": {"format": "lance"},
            "points": {"format": "lance"},
        },
    }
    path = BASE / "_gds_meta" / "sphere.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sphere, indent=2))


def generate_customers() -> None:
    schema = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("version", pa.int32()),
            pa.field("status", pa.string()),
            pa.field("name", pa.string()),
            pa.field("region", pa.string()),
            pa.field("balance", pa.float64()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("changed_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    table = pa.table(
        {
            "primary_key": ["CUST-001", "CUST-002", "CUST-003"],
            "version": [1, 1, 1],
            "status": ["active", "active", "expired"],
            "name": ["Acme Corp", "Bob Industries", "Charlie Ltd"],
            "region": ["EMEA", "EMEA", "APAC"],
            "balance": pa.array([9999.50, 150.75, -500.00], type=pa.float64()),
            "created_at": [ts, ts, ts],
            "changed_at": [ts, ts, ts],
        },
        schema=schema,
    )
    lance_path = BASE / "points" / "customers" / "v=1" / "data.lance"
    _write_lance(lance_path, table)
    # Build INVERTED indices on string columns for full-text search
    ds = lance.dataset(str(lance_path))
    for field in table.schema:
        if pa.types.is_string(field.type) and field.name != "primary_key":
            with contextlib.suppress(Exception):
                ds.create_scalar_index(field.name, index_type="INVERTED")


def generate_geometry() -> None:
    edge_struct_type = pa.struct(
        [
            pa.field("line_id", pa.string()),
            pa.field("point_key", pa.string()),
            pa.field("status", pa.string()),
            pa.field("direction", pa.string()),
        ]
    )
    schema = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("scale", pa.int32()),
            pa.field("delta", pa.list_(pa.float32())),
            pa.field("delta_norm", pa.float32()),
            pa.field("delta_rank_pct", pa.float32()),
            pa.field("is_anomaly", pa.bool_()),
            pa.field("edges", pa.list_(edge_struct_type)),
            pa.field("last_refresh_at", pa.timestamp("us", tz="UTC")),
            pa.field("updated_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    edges_001 = [
        {"line_id": "products", "point_key": "PROD-001", "status": "alive", "direction": "out"},
        {"line_id": "products", "point_key": "PROD-002", "status": "alive", "direction": "out"},
    ]
    edges_002 = [
        {"line_id": "products", "point_key": "PROD-003", "status": "alive", "direction": "out"},
    ]
    # Z-scored deltas: delta_z = (shape - mu) / sigma
    # theta_z = [7.5, 5.0], theta_z_norm = sqrt(81.25) ≈ 9.01
    # CUST-001: delta_z=[10.0, 5.0], norm≈11.18 > 9.01 → anomaly
    # CUST-002: delta_z=[1.5, -1.333], norm≈2.01 < 9.01 → normal
    delta_norms_raw = np.array([11.18, 2.01], dtype=np.float32)
    sorted_norms = np.sort(delta_norms_raw)
    ranks = np.searchsorted(sorted_norms, delta_norms_raw, side="right")
    delta_rank_pcts = (ranks / len(delta_norms_raw) * 100).astype(np.float32)
    table = pa.table(
        {
            "primary_key": ["CUST-001", "CUST-002"],
            "scale": [1, 1],
            "delta": [[10.0, 5.0], [1.5, -1.333333]],
            "delta_norm": delta_norms_raw.tolist(),
            "delta_rank_pct": delta_rank_pcts.tolist(),
            "is_anomaly": [True, False],
            "edges": pa.array([edges_001, edges_002], type=pa.list_(edge_struct_type)),
            "last_refresh_at": [ts, ts],
            "updated_at": [ts, ts],
        },
        schema=schema,
    )
    table = table.sort_by([("delta_norm", "descending")])
    # Per-dimension scalar columns for Lance predicate pushdown
    delta_col = table["delta"]
    list_size = len(delta_col[0].as_py())
    fixed_type = pa.list_(pa.float32(), list_size)
    fixed_delta = delta_col.cast(fixed_type)
    table = table.set_column(table.schema.get_field_index("delta"), "delta", fixed_delta)
    flat = fixed_delta.combine_chunks().values.to_numpy(zero_copy_only=False)
    matrix = flat.reshape(-1, list_size)
    for dim_idx in range(list_size):
        table = table.append_column(
            f"delta_dim_{dim_idx}",
            pa.array(matrix[:, dim_idx], type=pa.float32()),
        )
    _write_lance(BASE / "geometry" / "customer_pattern" / "v=1" / "data.lance", table)
    # Build geometry stats cache
    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(base_path=str(BASE))
    # theta: [7.5, 5.0] — matches customer_pattern in sphere.json
    theta_norm = float(np.linalg.norm(np.array([7.5, 5.0], dtype=np.float32)))
    writer.write_geometry_stats(
        "customer_pattern",
        version=1,
        delta_norms=delta_norms_raw.astype(np.float64),
        theta_norm=theta_norm,
    )


def generate_temporal() -> None:
    schema = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("slice_index", pa.int32()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("deformation_type", pa.string()),
            pa.field("shape_snapshot", pa.list_(pa.float32())),
            pa.field("pattern_ver", pa.int32()),
            pa.field("changed_property", pa.string()),
            pa.field("changed_line_id", pa.string()),
        ]
    )
    # customer_pattern: mu=[0.7, 0.4], sigma_diag=[0.2, 0.3]
    # shape = delta * sigma_diag + mu (reverse of z-scoring)
    # Original deltas: CUST-001: [-1.5,1.667],[-1.0,2.0],[-0.5,2.5]; CUST-002: [0.5,-0.5],[1.0,-1.0]
    # CUST-001: 3 slices — non-trivial mean and std for trajectory vector
    # CUST-002: 2 slices — minimum needed to compute displacement
    table = pa.table(
        {
            "primary_key": [
                "CUST-001",
                "CUST-001",
                "CUST-001",
                "CUST-002",
                "CUST-002",
            ],
            "slice_index": [0, 1, 2, 0, 1],
            "timestamp": [
                datetime(2023, 6, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 6, 1, tzinfo=UTC),
                datetime(2023, 9, 1, tzinfo=UTC),
                datetime(2024, 3, 1, tzinfo=UTC),
            ],
            "deformation_type": ["structural", "edge", "edge", "structural", "edge"],
            "shape_snapshot": [
                [0.4, 0.9001],
                [0.5, 1.0],
                [0.6, 1.15],
                [0.8, 0.25],
                [0.9, 0.1],
            ],
            "pattern_ver": [1, 1, 1, 1, 1],
            "changed_property": [None, None, None, None, None],
            "changed_line_id": [
                "products",
                "products",
                "products",
                "products",
                "products",
            ],
        },
        schema=schema,
    )
    _write_lance(BASE / "temporal" / "customer_pattern" / "data.lance", table)

    # Build trajectory ANN index for customer_pattern
    # Pass mu/sigma_diag so trajectory vectors use z-scored deltas
    mu = np.array([0.7, 0.4], dtype=np.float32)
    sigma_diag = np.array([0.2, 0.3], dtype=np.float32)
    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(base_path=str(BASE))
    writer.build_trajectory_index("customer_pattern", mu=mu, sigma_diag=sigma_diag)


if __name__ == "__main__":
    generate_sphere_json()
    generate_customers()
    generate_geometry()
    generate_temporal()
    print("Fixtures generated at", BASE)
