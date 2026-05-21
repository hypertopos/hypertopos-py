"""TDD tests for M3.1 find_topological_anomalies.

Test geometry per advisor 2026-05-11: tight 2D loop (30 pts, radius 0.3)
embedded at origin + Gaussian background mean=5 (≈17σ away on dim-0). For a
loop member, k=30 NN ≈ all 30 loop points (cycle visible in H_1). For a bg
member, k=30 NN = bg blob (no H_1). Top-n is dominated by loop members.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("ripser")
pytest.importorskip("sklearn")


def _embedded_loop_with_bg(
    n_loop: int = 30, n_bg: int = 1970, dim: int = 10, seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Loop members prefixed 'L' with LARGE cycle radius 3.0 → high h1_max;
    background prefixed 'B' as a tight Gaussian blob (std=0.3) far away → low h1_max.

    Ranking is by ``h1_max_persistence``: the loop's cycle lifetime (~3) dominates
    the tight background's local H_1 (~0.3). Earlier synthetic relied on the
    h1/h0 ratio favouring tight loops; that normalisation was empirically
    refuted on labelled fraud data so the engine now ranks on raw h1_max.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2 * np.pi, n_loop, endpoint=False)
    loop = np.zeros((n_loop, dim))
    loop[:, 0] = 3.0 * np.cos(theta) + rng.normal(0.0, 0.03, n_loop)
    loop[:, 1] = 3.0 * np.sin(theta) + rng.normal(0.0, 0.03, n_loop)
    bg = rng.normal(50.0, 0.3, (n_bg, dim))
    coords = np.vstack([loop, bg])
    pks = [f"L{i}" for i in range(n_loop)] + [f"B{i}" for i in range(n_bg)]
    return coords, pks


def _gaussian_cloud(n: int, dim: int = 10, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (n, dim))


def _to_geometry_table(coords: np.ndarray, pks: list[str] | None = None) -> pa.Table:
    n, d = coords.shape
    if pks is None:
        pks = [f"e{i}" for i in range(n)]
    cols = {"primary_key": pa.array(pks, type=pa.string())}
    for j in range(d):
        cols[f"d{j}"] = pa.array(coords[:, j].astype(np.float64), type=pa.float64())
    return pa.table(cols)


def test_embedded_loop_dominates_top_n():
    from hypertopos.engine.topology import find_topological_anomalies

    coords, pks = _embedded_loop_with_bg(n_loop=30, n_bg=1470, dim=10, seed=0)
    tbl = _to_geometry_table(coords, pks)

    result = find_topological_anomalies(
        tbl, top_n=30, sample_size=1500, k_neighbors=30, pca_dim=10,
    )

    top_pks = {r["primary_key"] for r in result}
    loop_pks = {f"L{i}" for i in range(30)}
    overlap = len(top_pks & loop_pks)
    assert overlap >= 20, (
        f"only {overlap}/30 top-30 entities are loop members "
        f"(L*); top primary_keys: {sorted(top_pks)[:10]}"
    )


def test_below_min_n_raises():
    from hypertopos.engine.topology import find_topological_anomalies

    tbl = _to_geometry_table(_gaussian_cloud(500))
    with pytest.raises(ValueError, match="n_entities"):
        find_topological_anomalies(tbl, top_n=20, sample_size=500)


def test_advisory_warning_for_small_n():
    from hypertopos.engine.topology import find_topological_anomalies

    tbl = _to_geometry_table(_gaussian_cloud(1500))
    with pytest.warns(UserWarning, match="PH reliability"):
        find_topological_anomalies(
            tbl, top_n=10, sample_size=1500, k_neighbors=20,
        )


def test_non_numeric_columns_filtered_silently():
    from hypertopos.engine.topology import find_topological_anomalies

    coords, pks = _embedded_loop_with_bg(n_loop=20, n_bg=1480, dim=8, seed=3)
    cols: dict[str, pa.Array] = {
        "primary_key": pa.array(pks, type=pa.string()),
        "label": pa.array(["x"] * len(pks), type=pa.string()),
        "event_ids": pa.array([[1, 2]] * len(pks), type=pa.list_(pa.int64())),
    }
    for j in range(8):
        cols[f"d{j}"] = pa.array(coords[:, j].astype(np.float64), type=pa.float64())
    tbl = pa.table(cols)

    result = find_topological_anomalies(
        tbl, top_n=10, sample_size=1500, k_neighbors=20,
    )
    assert len(result) == 10
    assert all(np.isfinite(r["topo_score"]) for r in result)


def test_return_shape_and_fields():
    from hypertopos.engine.topology import find_topological_anomalies

    coords, pks = _embedded_loop_with_bg(n_loop=20, n_bg=1480, dim=10, seed=2)
    tbl = _to_geometry_table(coords, pks)

    result = find_topological_anomalies(
        tbl, top_n=10, sample_size=1500, k_neighbors=20,
    )

    assert isinstance(result, list)
    assert len(result) == 10
    required = {
        "primary_key", "topo_score", "h1_max_persistence",
        "h0_mean_death", "n_h1_features", "computed_at",
    }
    for row in result:
        assert required.issubset(row.keys()), f"missing fields: {required - row.keys()}"
        assert isinstance(row["primary_key"], str)
        assert np.isfinite(row["topo_score"])
        assert row["topo_score"] >= 0.0
