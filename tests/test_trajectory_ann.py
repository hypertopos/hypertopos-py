# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Trajectory ANN index — π10_attract_trajectory (MCP tool find_drifting_similar)."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import lance as _lance
import numpy as np
import pytest

from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter
from tests.conftest import FIXTURES_PATH, clone_sphere


@pytest.fixture(scope="module")
def fixture_sphere_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-private clone: these tests rebuild the trajectory index, which
    must not commit into the shared fixture other workers read."""
    return clone_sphere(FIXTURES_PATH, tmp_path_factory.mktemp("trajectory_ann") / "sphere")


def _build_nav(fixture_sphere_path: Path):
    from hypertopos.engine.geometry import GDSEngine
    from hypertopos.model.manifest import Contract, Manifest
    from hypertopos.navigation.navigator import GDSNavigator
    from hypertopos.storage.cache import GDSCache

    reader = GDSReader(str(fixture_sphere_path))
    sphere = reader.read_sphere()
    cache = GDSCache()
    engine = GDSEngine(storage=reader, cache=cache)
    manifest = Manifest(
        manifest_id=str(uuid.uuid4()),
        agent_id="test",
        snapshot_time=datetime.now(tz=UTC),
        status="active",
        line_versions={lid: line.current_version() for lid, line in sphere.lines.items()},
        pattern_versions={pid: p.version for pid, p in sphere.patterns.items()},
    )
    contract = Contract(
        manifest_id=manifest.manifest_id,
        pattern_ids=list(sphere.patterns.keys()),
    )
    return GDSNavigator(engine=engine, storage=reader, manifest=manifest, contract=contract)


def _get_anchor_pattern_with_temporal(fixture_sphere_path: Path) -> str | None:
    sphere_json = json.loads((fixture_sphere_path / "_gds_meta" / "sphere.json").read_text())
    patterns = sphere_json["patterns"]
    for k, v in patterns.items():
        if (
            v.get("pattern_type") == "anchor"
            and (fixture_sphere_path / "temporal" / k / "data.lance").exists()
        ):
            return k
    return None


def test_build_trajectory_index_creates_lance(fixture_sphere_path: Path) -> None:
    pattern_id = _get_anchor_pattern_with_temporal(fixture_sphere_path)
    if pattern_id is None:
        pytest.skip("No anchor pattern with temporal data in fixture")
    writer = GDSWriter(str(fixture_sphere_path))
    n = writer.build_trajectory_index(pattern_id)
    assert n > 0
    traj_path = fixture_sphere_path / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    assert traj_path.exists()


def test_trajectory_index_schema(fixture_sphere_path: Path) -> None:
    pattern_id = _get_anchor_pattern_with_temporal(fixture_sphere_path)
    if pattern_id is None:
        pytest.skip("No anchor pattern with temporal data in fixture")
    writer = GDSWriter(str(fixture_sphere_path))
    writer.build_trajectory_index(pattern_id)
    traj_path = fixture_sphere_path / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    ds = _lance.dataset(str(traj_path))
    names = [f.name for f in ds.schema]
    assert "primary_key" in names
    assert "trajectory_vector" in names
    assert "displacement" in names
    assert "num_slices" in names
    assert "first_timestamp" in names
    assert "last_timestamp" in names


def test_find_drifting_similar_returns_results(fixture_sphere_path: Path) -> None:
    pattern_id = _get_anchor_pattern_with_temporal(fixture_sphere_path)
    if pattern_id is None:
        pytest.skip("No anchor pattern with temporal data in fixture")
    writer = GDSWriter(str(fixture_sphere_path))
    writer.build_trajectory_index(pattern_id)

    nav = _build_nav(fixture_sphere_path)

    traj_path = fixture_sphere_path / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    ds = _lance.dataset(str(traj_path))
    sample = ds.to_table(limit=1, columns=["primary_key"])
    if sample.num_rows == 0:
        pytest.skip("Trajectory index empty")
    query_pk = sample["primary_key"][0].as_py()

    results = nav.π10_attract_trajectory(query_pk, pattern_id, top_n=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    assert all("primary_key" in r and "distance" in r for r in results)
    assert all(r["primary_key"] != query_pk for r in results)  # excludes self


def test_find_drifting_similar_raises_for_event_pattern(
    fixture_sphere_path: Path, tmp_path: Path
) -> None:
    """The fixture has no event pattern: a sphere.json-only copy relabels one."""
    sphere_json = json.loads((fixture_sphere_path / "_gds_meta" / "sphere.json").read_text())
    event_pid = next(iter(sphere_json["patterns"]))
    sphere_json["patterns"][event_pid]["pattern_type"] = "event"
    meta = tmp_path / "sphere" / "_gds_meta"
    meta.mkdir(parents=True)
    (meta / "sphere.json").write_text(json.dumps(sphere_json))
    nav = _build_nav(tmp_path / "sphere")
    with pytest.raises(ValueError, match="anchor"):
        nav.π10_attract_trajectory("KEY-001", event_pid, top_n=3)


def test_build_trajectory_index_returns_zero_without_temporal(
    fixture_sphere_path: Path,
) -> None:
    writer = GDSWriter(str(fixture_sphere_path))
    n = writer.build_trajectory_index("nonexistent_pattern_xyz")
    assert n == 0


def test_find_drifting_similar_raises_for_single_slice_entity(
    fixture_sphere_path: Path, tmp_path: Path
) -> None:
    """Raises ValueError when reference entity has only 1 temporal slice."""
    import shutil

    import pyarrow as pa

    pattern_id = _get_anchor_pattern_with_temporal(fixture_sphere_path)
    if pattern_id is None:
        pytest.skip("No anchor pattern with temporal data in fixture")

    # Build trajectory index on shared fixture (idempotent) to get vector dim
    writer = GDSWriter(str(fixture_sphere_path))
    writer.build_trajectory_index(pattern_id)
    src_traj = fixture_sphere_path / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    ds = _lance.dataset(str(src_traj))
    dim = len(ds.to_table(limit=1)["trajectory_vector"][0].as_py())

    # Create minimal sphere in tmp_path: only sphere.json + synthetic trajectory
    mini = tmp_path / "sphere"
    meta = mini / "_gds_meta"
    meta.mkdir(parents=True)
    shutil.copy2(
        str(fixture_sphere_path / "_gds_meta" / "sphere.json"),
        str(meta / "sphere.json"),
    )

    test_key = "__single_slice_test__"
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    traj_table = pa.table(
        {
            "primary_key": [test_key],
            "trajectory_vector": pa.array([[0.0] * dim], type=pa.list_(pa.float32(), dim)),
            "displacement": pa.array([0.0], type=pa.float32()),
            "num_slices": pa.array([1], type=pa.int32()),
            "first_timestamp": pa.array([ts], type=pa.timestamp("us", tz="UTC")),
            "last_timestamp": pa.array([ts], type=pa.timestamp("us", tz="UTC")),
        }
    )
    traj_dir = meta / "trajectory"
    traj_dir.mkdir()
    _lance.write_dataset(traj_table, str(traj_dir / f"{pattern_id}.lance"))

    nav = _build_nav(mini)
    with pytest.raises(ValueError, match="insufficient"):
        nav.π10_attract_trajectory(test_key, pattern_id, top_n=3)


def test_find_drifting_similar_raises_without_index(
    fixture_sphere_path: Path, tmp_path: Path
) -> None:
    """Raises ValueError when trajectory index does not exist."""
    import shutil

    # Create minimal sphere in tmp_path with only sphere.json — no trajectory index
    mini = tmp_path / "sphere"
    meta = mini / "_gds_meta"
    meta.mkdir(parents=True)
    shutil.copy2(
        str(fixture_sphere_path / "_gds_meta" / "sphere.json"),
        str(meta / "sphere.json"),
    )

    nav = _build_nav(mini)
    with pytest.raises(ValueError, match="Trajectory index not found"):
        nav.π10_attract_trajectory("CUST-001", "customer_pattern", top_n=3)


def test_build_trajectory_uses_ivf_flat(
    fixture_sphere_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Index type should be IVF_FLAT, not IVF_PQ.

    The fixture population is below the real IVF threshold, so the partition
    count and per-partition minimum are lowered to let the index be trained.
    """
    import shutil

    import hypertopos.storage.writer as writer_mod

    pattern_id = _get_anchor_pattern_with_temporal(fixture_sphere_path)
    if pattern_id is None:
        pytest.skip("No anchor pattern with temporal data in fixture")
    sphere = tmp_path / "sphere"
    shutil.copytree(fixture_sphere_path, sphere)
    monkeypatch.setattr(writer_mod, "_ivf_num_partitions", lambda _n: 2)
    monkeypatch.setattr(writer_mod, "_IVF_MIN_ROWS_PER_PARTITION", 1)
    n = GDSWriter(str(sphere)).build_trajectory_index(pattern_id)
    assert writer_mod._ivf_index_worthwhile(n)
    traj_path = sphere / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    ds = _lance.dataset(str(traj_path))
    indices = ds.describe_indices()
    assert len(indices) > 0
    idx = indices[0]
    index_type = ds.stats.index_stats(idx.name)["index_type"]
    assert index_type == "IVF_FLAT", f"Expected IVF_FLAT index, got {index_type}"


def test_trajectory_vectorized_correctness(
    fixture_sphere_path: Path,
) -> None:
    """Vectorized trajectory equals mean/std of the per-entity pattern-metric deltas
    of its active slices."""
    from hypertopos.engine.geometry import slice_deltas
    from hypertopos.navigation.navigator import active_slice_mask

    pattern_id = _get_anchor_pattern_with_temporal(fixture_sphere_path)
    if pattern_id is None:
        pytest.skip("No anchor pattern with temporal data in fixture")

    writer = GDSWriter(str(fixture_sphere_path))
    writer.build_trajectory_index(pattern_id)

    traj_path = fixture_sphere_path / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    ds = _lance.dataset(str(traj_path))
    traj_table = ds.to_table()

    temporal_path = fixture_sphere_path / "temporal" / pattern_id / "data.lance"
    temp_ds = _lance.dataset(str(temporal_path))
    temp_table = temp_ds.to_table(columns=["primary_key", "shape_snapshot", "timestamp"])

    for i in range(min(3, traj_table.num_rows)):
        pk = traj_table["primary_key"][i].as_py()
        import pyarrow.compute as pc

        mask = pc.equal(temp_table["primary_key"], pk)
        entity_rows = temp_table.filter(mask)
        sort_idx = pc.sort_indices(entity_rows, sort_keys=[("timestamp", "ascending")])
        entity_rows = entity_rows.take(sort_idx)

        shapes = []
        for j in range(entity_rows.num_rows):
            shapes.append(entity_rows["shape_snapshot"][j].as_py())
        pattern = GDSReader(str(fixture_sphere_path)).read_sphere().patterns[pattern_id]
        raw = np.array(shapes, dtype=np.float32)
        shapes = slice_deltas(raw[active_slice_mask(raw)], pattern)  # active windows only

        expected_mean = shapes.mean(axis=0)
        expected_std = shapes.std(axis=0) if len(shapes) > 1 else np.zeros_like(expected_mean)

        actual_vec = np.array(traj_table["trajectory_vector"][i].as_py(), dtype=np.float32)
        D = len(expected_mean)
        actual_mean = actual_vec[:D]
        actual_std = actual_vec[D:]

        np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(actual_std, expected_std, rtol=1e-4, atol=1e-6)


def _make_subthreshold_traj_sphere(
    fixture_sphere_path: Path,
    dst: Path,
    n_entities: int,
) -> tuple[str, list[str]]:
    """Build a minimal sphere whose trajectory dataset is below the IVF
    KMeans training threshold, written via the real build path
    (``write_trajectory_from_tensor``).

    Returns (pattern_id, ordered primary keys). Entity i sits at mean shape
    [i, 0] held constant across buckets (std == 0), so nearest-neighbour in
    trajectory-vector space is deterministic: entity i's nearest is i±1.
    """
    import shutil

    pattern_id = "customer_pattern"
    meta = dst / "_gds_meta"
    meta.mkdir(parents=True)
    shutil.copy2(
        str(fixture_sphere_path / "_gds_meta" / "sphere.json"),
        str(meta / "sphere.json"),
    )

    n_buckets = 3
    d = 2
    shape_tensor = np.zeros((n_entities, n_buckets, d), dtype=np.float32)
    for i in range(n_entities):
        shape_tensor[i, :, 0] = float(i)  # distinct mean per entity, std 0
    pks = [f"E-{i:03d}" for i in range(n_entities)]
    bucket_ts = [datetime(2024, 1, 1 + b, tzinfo=UTC) for b in range(n_buckets)]

    writer = GDSWriter(str(dst))
    n = writer.write_trajectory_from_tensor(
        pattern_id,
        shape_tensor,
        pks,
        bucket_timestamps=bucket_ts,
        raw_shapes=shape_tensor,
    )
    assert n == n_entities
    return pattern_id, pks


def test_subthreshold_trajectory_skips_ivf_index(fixture_sphere_path: Path, tmp_path: Path) -> None:
    """Below the KMeans training threshold, no IVF index is created."""
    from hypertopos.storage.writer import _ivf_index_worthwhile

    n_entities = 50  # << num_partitions * 256 (>= 4096)
    assert not _ivf_index_worthwhile(n_entities)

    mini = tmp_path / "sphere"
    pattern_id, _pks = _make_subthreshold_traj_sphere(
        fixture_sphere_path,
        mini,
        n_entities,
    )

    traj_path = mini / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    assert traj_path.exists()  # dataset is written — only the index is skipped
    ds = _lance.dataset(str(traj_path))
    assert ds.count_rows() == n_entities
    assert ds.describe_indices() == [], (
        "sub-threshold trajectory dataset must carry no IVF vector index"
    )


def test_subthreshold_trajectory_query_uses_full_scan_fallback(
    fixture_sphere_path: Path, tmp_path: Path
) -> None:
    """Without an IVF index, π10 still returns the geometrically correct
    nearest trajectory via Lance's brute-force scan."""
    n_entities = 50
    mini = tmp_path / "sphere"
    pattern_id, pks = _make_subthreshold_traj_sphere(
        fixture_sphere_path,
        mini,
        n_entities,
    )

    traj_path = mini / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
    assert _lance.dataset(str(traj_path)).describe_indices() == []

    nav = _build_nav(mini)

    # Query a middle entity: its two equidistant nearest neighbours are i±1.
    query_idx = 25
    results = nav.π10_attract_trajectory(pks[query_idx], pattern_id, top_n=3)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(r["primary_key"] != pks[query_idx] for r in results)  # excludes self
    # Closest by Euclidean distance over [mean, std] is the adjacent index.
    nearest = {pks[query_idx - 1], pks[query_idx + 1]}
    assert results[0]["primary_key"] in nearest, (
        f"flat-scan nearest should be an adjacent entity, got {results[0]['primary_key']}"
    )
    # Distances are sorted ascending and match the known geometry (|Δmean|=1).
    assert results[0]["distance"] == pytest.approx(1.0, abs=1e-4)
