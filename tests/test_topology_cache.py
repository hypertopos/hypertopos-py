"""Cache lifecycle for topology sidecar Lance store."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

from hypertopos.storage.topology_cache import (
    ANOMALIES_SCHEMA,
    TRAJECTORY_SCHEMA,
    anomalies_params_key,
    cache_path,
    read_cache,
    write_cache,
)


def test_cache_path_layout(tmp_path: Path):
    p = cache_path(tmp_path, "anomalies", "account_pattern", 3)
    assert p == tmp_path / "_gds_meta" / "topology_cache" / "anomalies" / "account_pattern" / "v=3.lance"


def test_cache_path_params_key_distinct(tmp_path: Path):
    """Distinct scoring params get distinct cache files at the same version;
    the empty default stays back-compatible with the bare v=N.lance name."""
    base = cache_path(tmp_path, "anomalies", "acct", 1)
    k50 = cache_path(
        tmp_path, "anomalies", "acct", 1,
        anomalies_params_key(
            k_neighbors=50, pca_dim=10, sample_size=50_000, homology_dim=1,
        ),
    )
    k5 = cache_path(
        tmp_path, "anomalies", "acct", 1,
        anomalies_params_key(
            k_neighbors=5, pca_dim=10, sample_size=50_000, homology_dim=1,
        ),
    )
    assert base.name == "v=1.lance"
    assert k50.name == "v=1_k50_pca10_s50000_h1.lance"
    assert k50 != k5 and k50 != base and k5 != base


def test_read_returns_none_when_missing(tmp_path: Path):
    p = cache_path(tmp_path, "anomalies", "acct", 1)
    assert read_cache(p) is None


def test_round_trip_anomalies(tmp_path: Path):
    p = cache_path(tmp_path, "anomalies", "acct", 1)
    rows = [
        {
            "primary_key": f"e{i}",
            "topo_score": float(i),
            "h1_max_persistence": float(i) / 10.0,
            "h0_mean_death": 0.5,
            "n_h1_features": i,
            "computed_at": datetime.now(timezone.utc),
        }
        for i in range(5)
    ]
    write_cache(p, rows, ANOMALIES_SCHEMA)
    tbl = read_cache(p)
    assert tbl is not None
    assert tbl.num_rows == 5
    assert tbl.schema.field("topo_score").type == pa.float64()
    assert tbl["primary_key"].to_pylist() == [f"e{i}" for i in range(5)]


def test_round_trip_trajectory(tmp_path: Path):
    p = cache_path(tmp_path, "trajectory", "acct", 2)
    rows = [
        {
            "primary_key": "e0",
            "trajectory_topo_score": 1.5,
            "n_timesteps": 20,
            "h1_total_persistence": 1.5,
            "dominant_feature_birth": 0.1,
            "dominant_feature_death": 0.6,
            "computed_at": datetime.now(timezone.utc),
        },
    ]
    write_cache(p, rows, TRAJECTORY_SCHEMA)
    tbl = read_cache(p)
    assert tbl is not None
    assert tbl.num_rows == 1
    assert tbl.schema.field("n_timesteps").type == pa.int32()


def test_overwrite_replaces_existing(tmp_path: Path):
    p = cache_path(tmp_path, "anomalies", "acct", 1)
    now = datetime.now(timezone.utc)
    write_cache(p, [{
        "primary_key": "a", "topo_score": 1.0, "h1_max_persistence": 0.1,
        "h0_mean_death": 0.5, "n_h1_features": 1, "computed_at": now,
    }], ANOMALIES_SCHEMA)
    write_cache(p, [
        {
            "primary_key": "b", "topo_score": 2.0, "h1_max_persistence": 0.2,
            "h0_mean_death": 0.5, "n_h1_features": 2, "computed_at": now,
        },
        {
            "primary_key": "c", "topo_score": 3.0, "h1_max_persistence": 0.3,
            "h0_mean_death": 0.5, "n_h1_features": 3, "computed_at": now,
        },
    ], ANOMALIES_SCHEMA)
    tbl = read_cache(p)
    assert tbl is not None
    assert tbl.num_rows == 2
    assert set(tbl["primary_key"].to_pylist()) == {"b", "c"}


def test_version_isolation(tmp_path: Path):
    p1 = cache_path(tmp_path, "anomalies", "acct", 1)
    p2 = cache_path(tmp_path, "anomalies", "acct", 2)
    now = datetime.now(timezone.utc)
    write_cache(p1, [{
        "primary_key": "v1_row", "topo_score": 1.0, "h1_max_persistence": 0.0,
        "h0_mean_death": 0.0, "n_h1_features": 0, "computed_at": now,
    }], ANOMALIES_SCHEMA)
    write_cache(p2, [{
        "primary_key": "v2_row", "topo_score": 2.0, "h1_max_persistence": 0.0,
        "h0_mean_death": 0.0, "n_h1_features": 0, "computed_at": now,
    }], ANOMALIES_SCHEMA)
    t1 = read_cache(p1)
    t2 = read_cache(p2)
    assert t1 is not None and t1["primary_key"].to_pylist() == ["v1_row"]
    assert t2 is not None and t2["primary_key"].to_pylist() == ["v2_row"]
