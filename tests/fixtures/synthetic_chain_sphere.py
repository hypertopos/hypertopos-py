"""Synthetic chain-anchor sphere fixture for chain-coherent unit tests.

Hand-crafted minimal sphere that exercises the chain investigative loop
primitives (`find_chains_with_coherent_anomaly`, `anomaly_propagation_in_
chain`, `classify_chain_typology`, `extend_chain`, `find_chains_for_entity`)
without depending on the AML HI-small benchmark sphere build.

Sphere shape:
- accounts line — 8 entities (A1..A4 anomalous on dim 0, B1..B4 clean)
- chains line — 12 chains exercising different chain-coherent topologies
  (full cascade, partial cascade, clean chain, interleaved, cyclic, self-
  loop, short, long-with-reset).
- account_pattern — anchor over accounts, 3 dims with hand-set deltas
- chain_pattern — anchor over chains, 2 dims with hand-set deltas

Compact enough to build in <50 ms, large enough to give every chain
primitive a non-trivial input shape. Live alongside the AML HI-small
fixture in conftest as an opt-in session fixture.
"""
from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

BASE = Path(__file__).parent / "gds" / "synthetic_chain_sphere"

# Account anomaly geometry — A1..A4 anomalous on dim_0, sharing top_dim.
# B1..B4 are clean.
ACCOUNT_DIMS = 3
ACCOUNT_MU = np.array([0.5, 0.5, 0.5], dtype=np.float32)
ACCOUNT_SIGMA = np.array([0.1, 0.1, 0.1], dtype=np.float32)
ACCOUNT_THETA = np.array([3.0, 3.0, 3.0], dtype=np.float32)

ACCOUNT_DELTAS = {
    "A1": [5.0, 0.5, 0.3],
    "A2": [6.0, 0.3, 0.4],
    "A3": [4.5, 0.1, 0.5],
    "A4": [5.5, 0.4, 0.2],
    "B1": [0.4, 0.3, 0.2],
    "B2": [0.5, 0.4, 0.3],
    "B3": [0.3, 0.2, 0.4],
    "B4": [0.2, 0.5, 0.3],
}

# Chain shape — 12 chains exercising different topologies.
CHAINS = {
    "CH-001": ["A1", "A2", "A3", "A4"],          # full cascade (n=4)
    "CH-002": ["A1", "A2", "B1"],                  # partial cascade (anom prefix)
    "CH-003": ["B1", "B2", "B3"],                  # clean chain
    "CH-004": ["A1", "B1", "A2"],                  # interleaved
    "CH-005": ["A1", "A2"],                         # short cascade (n=2)
    "CH-006": ["A3", "A4", "A1"],                  # different cascade ordering
    "CH-007": ["A2", "B2"],                         # boundary 1+1
    "CH-008": ["A1", "A1"],                         # self-loop
    "CH-009": ["A1", "A2", "B1", "A3", "A4"],   # reset-then-resume
    "CH-010": ["B3", "B4"],                         # clean short
    "CH-011": ["A1", "A2", "A3", "A4", "B4"],   # cascade then clean exit
    "CH-012": ["B1", "A1", "A2", "A3"],           # clean-then-cascade
}

# Chain pattern level — 2 dims (synthetic chain features).
CHAIN_DIMS = 2
CHAIN_MU = np.array([2.5, 1.0], dtype=np.float32)
CHAIN_SIGMA = np.array([1.0, 0.5], dtype=np.float32)
CHAIN_THETA = np.array([3.0, 3.0], dtype=np.float32)


def _write_lance(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(path), mode="overwrite")


def _attach_per_dim_columns(table: pa.Table, list_size: int) -> pa.Table:
    """Cast `delta` to fixed-size list and attach delta_dim_N columns."""
    delta_col = table["delta"]
    fixed_type = pa.list_(pa.float32(), list_size)
    fixed_delta = delta_col.cast(fixed_type)
    table = table.set_column(
        table.schema.get_field_index("delta"), "delta", fixed_delta,
    )
    flat = fixed_delta.combine_chunks().values.to_numpy(zero_copy_only=False)
    matrix = flat.reshape(-1, list_size)
    for i in range(list_size):
        table = table.append_column(
            f"delta_dim_{i}",
            pa.array(matrix[:, i], type=pa.float32()),
        )
    return table


def generate_sphere_json() -> None:
    sphere = {
        "sphere_id": "synthetic_chain_sphere",
        "name": "Synthetic Chain Sphere",
        "lines": {
            "accounts": {
                "line_id": "accounts",
                "entity_type": "account",
                "line_role": "anchor",
                "pattern_id": "account_pattern",
                "partitioning": {"mode": "static", "columns": []},
                "versions": [1],
                "columns": [{"name": "primary_key", "type": "string"}],
            },
            "chains": {
                "line_id": "chains",
                "entity_type": "chain",
                "line_role": "anchor",
                "pattern_id": "chain_pattern",
                "partitioning": {"mode": "static", "columns": []},
                "versions": [1],
                "columns": [
                    {"name": "primary_key", "type": "string"},
                    {"name": "chain_keys", "type": "string"},
                    {"name": "hop_count", "type": "int32"},
                    {"name": "is_cyclic", "type": "bool"},
                ],
            },
        },
        "patterns": {
            "account_pattern": {
                "pattern_id": "account_pattern",
                "entity_type": "account",
                "pattern_type": "anchor",
                "version": 1,
                "status": "production",
                "relations": [],
                "mu": ACCOUNT_MU.tolist(),
                "sigma_diag": ACCOUNT_SIGMA.tolist(),
                "theta": ACCOUNT_THETA.tolist(),
                "dim_labels": ["risk_score", "diversity", "regularity"],
                "population_size": len(ACCOUNT_DELTAS),
                "computed_at": "2026-05-08T00:00:00+00:00",
            },
            "chain_pattern": {
                "pattern_id": "chain_pattern",
                "entity_type": "chain",
                "pattern_type": "anchor",
                "version": 1,
                "status": "production",
                "relations": [],
                "mu": CHAIN_MU.tolist(),
                "sigma_diag": CHAIN_SIGMA.tolist(),
                "theta": CHAIN_THETA.tolist(),
                "dim_labels": ["hop_count", "time_span_hours"],
                "population_size": len(CHAINS),
                "computed_at": "2026-05-08T00:00:00+00:00",
            },
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


def generate_accounts_points() -> None:
    schema = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("version", pa.int32()),
            pa.field("status", pa.string()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("changed_at", pa.timestamp("us", tz="UTC")),
        ],
    )
    ts = datetime(2026, 5, 8, tzinfo=UTC)
    pks = list(ACCOUNT_DELTAS.keys())
    n = len(pks)
    table = pa.table(
        {
            "primary_key": pks,
            "version": [1] * n,
            "status": ["active"] * n,
            "created_at": [ts] * n,
            "changed_at": [ts] * n,
        },
        schema=schema,
    )
    _write_lance(BASE / "points" / "accounts" / "v=1" / "data.lance", table)


def generate_chains_points() -> None:
    schema = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("chain_keys", pa.string()),
            pa.field("hop_count", pa.int32()),
            pa.field("is_cyclic", pa.bool_()),
            pa.field("version", pa.int32()),
            pa.field("status", pa.string()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("changed_at", pa.timestamp("us", tz="UTC")),
        ],
    )
    ts = datetime(2026, 5, 8, tzinfo=UTC)
    pks = list(CHAINS.keys())
    keys_csv = [",".join(CHAINS[pk]) for pk in pks]
    hop_counts = [len(CHAINS[pk]) - 1 for pk in pks]
    is_cyclic = [
        len(CHAINS[pk]) >= 2 and CHAINS[pk][0] == CHAINS[pk][-1]
        for pk in pks
    ]
    n = len(pks)
    table = pa.table(
        {
            "primary_key": pks,
            "chain_keys": keys_csv,
            "hop_count": pa.array(hop_counts, type=pa.int32()),
            "is_cyclic": is_cyclic,
            "version": [1] * n,
            "status": ["active"] * n,
            "created_at": [ts] * n,
            "changed_at": [ts] * n,
        },
        schema=schema,
    )
    _write_lance(BASE / "points" / "chains" / "v=1" / "data.lance", table)


def _build_geometry_table(
    pks: list[str],
    deltas: dict[str, list[float]],
    mu: np.ndarray,
    sigma: np.ndarray,
    theta: np.ndarray,
    *,
    pattern_name: str,
) -> pa.Table:
    """Build a geometry table with calibrated z-scored deltas + is_anomaly."""
    edge_struct = pa.struct(
        [
            pa.field("line_id", pa.string()),
            pa.field("point_key", pa.string()),
            pa.field("status", pa.string()),
            pa.field("direction", pa.string()),
        ],
    )
    schema = pa.schema(
        [
            pa.field("primary_key", pa.string()),
            pa.field("scale", pa.int32()),
            pa.field("delta", pa.list_(pa.float32())),
            pa.field("delta_norm", pa.float32()),
            pa.field("delta_rank_pct", pa.float32()),
            pa.field("is_anomaly", pa.bool_()),
            pa.field("edges", pa.list_(edge_struct)),
            pa.field("last_refresh_at", pa.timestamp("us", tz="UTC")),
            pa.field("updated_at", pa.timestamp("us", tz="UTC")),
        ],
    )
    ts = datetime(2026, 5, 8, tzinfo=UTC)

    z_deltas = []
    norms = []
    for pk in pks:
        raw = np.array(deltas[pk], dtype=np.float32)
        z = (raw - mu) / sigma
        z_deltas.append(z.tolist())
        norms.append(float(np.linalg.norm(z)))

    norms_arr = np.array(norms, dtype=np.float32)
    sorted_norms = np.sort(norms_arr)
    ranks = np.searchsorted(sorted_norms, norms_arr, side="right")
    rank_pcts = (ranks / max(len(norms_arr), 1) * 100.0).astype(np.float32)
    theta_norm_z = float(np.linalg.norm(theta))
    is_anomaly = (norms_arr >= theta_norm_z).tolist()

    table = pa.table(
        {
            "primary_key": pks,
            "scale": [1] * len(pks),
            "delta": z_deltas,
            "delta_norm": norms_arr.tolist(),
            "delta_rank_pct": rank_pcts.tolist(),
            "is_anomaly": is_anomaly,
            "edges": pa.array([[]] * len(pks), type=pa.list_(edge_struct)),
            "last_refresh_at": [ts] * len(pks),
            "updated_at": [ts] * len(pks),
        },
        schema=schema,
    )
    table = table.sort_by([("delta_norm", "descending")])
    table = _attach_per_dim_columns(table, list_size=len(mu))
    _ = pattern_name  # kept for symmetry / potential downstream tagging
    return table


def generate_account_geometry() -> None:
    pks = list(ACCOUNT_DELTAS.keys())
    table = _build_geometry_table(
        pks, ACCOUNT_DELTAS, ACCOUNT_MU, ACCOUNT_SIGMA, ACCOUNT_THETA,
        pattern_name="account_pattern",
    )
    _write_lance(
        BASE / "geometry" / "account_pattern" / "v=1" / "data.lance", table,
    )

    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(base_path=str(BASE))
    norms = table["delta_norm"].to_numpy(zero_copy_only=False).astype(np.float64)
    theta_norm_z = float(np.linalg.norm(ACCOUNT_THETA / ACCOUNT_SIGMA))
    writer.write_geometry_stats(
        "account_pattern",
        version=1,
        delta_norms=norms,
        theta_norm=theta_norm_z,
    )


def generate_chain_geometry() -> None:
    """Build chain pattern geometry — chain-level deltas computed from
    hop_count + time_span (hand-crafted as derived from the chain shape).
    """
    pks = list(CHAINS.keys())
    chain_deltas = {}
    for pk in pks:
        n_hops = len(CHAINS[pk]) - 1
        # Chain-level dims [hop_count, time_span_hours]; fabricate
        # time_span proportional to n_hops with small jitter.
        chain_deltas[pk] = [float(n_hops), float(n_hops) * 0.5 + 0.5]

    table = _build_geometry_table(
        pks, chain_deltas, CHAIN_MU, CHAIN_SIGMA, CHAIN_THETA,
        pattern_name="chain_pattern",
    )
    _write_lance(
        BASE / "geometry" / "chain_pattern" / "v=1" / "data.lance", table,
    )

    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(base_path=str(BASE))
    norms = table["delta_norm"].to_numpy(zero_copy_only=False).astype(np.float64)
    theta_norm_z = float(np.linalg.norm(CHAIN_THETA / CHAIN_SIGMA))
    writer.write_geometry_stats(
        "chain_pattern",
        version=1,
        delta_norms=norms,
        theta_norm=theta_norm_z,
    )


def generate_all() -> None:
    generate_sphere_json()
    generate_accounts_points()
    generate_chains_points()
    generate_account_geometry()
    generate_chain_geometry()


if __name__ == "__main__":
    generate_all()
