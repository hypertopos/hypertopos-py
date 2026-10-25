"""Build-time trajectory vectors use only an entity's active windows, like π9 drift."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

_BUSY = [f"A-{i}" for i in range(1, 6)]
_LATE = "LATE"  # silent on days 1-3, active on days 4-5
_ONCE = "ONCE"  # a single active window
_DAYS = (1, 2, 3, 4, 5)


def _build(path: Path) -> Path:
    from hypertopos.builder import GDSBuilder

    rows = []
    for day in _DAYS:
        for n, acct in enumerate(_BUSY):
            for j in range(day % 3 + 1 + n % 2):
                rows.append((acct, float((day * 7 + n * 3 + j) % 13 + 1), day))
        if day >= 4:
            rows.extend([(_LATE, 40.0 * day, day), (_LATE, 5.0, day)])
        if day == 5:
            rows.append((_ONCE, 9.0, day))
    table = pa.table(
        {
            "tx_id": [f"TX-{i}" for i in range(len(rows))],
            "from_account": [r[0] for r in rows],
            "amount": [r[1] for r in rows],
            "timestamp": [datetime(2024, 1, r[2], 12) for r in rows],
        }
    )
    b = GDSBuilder("test", str(path))
    b.add_line("transactions", table, key_col="tx_id", source_id="test", role="event")
    b.add_line(
        "accounts",
        pa.table({"primary_key": [*_BUSY, _LATE, _ONCE]}),
        key_col="primary_key",
        source_id="test",
        role="anchor",
    )
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "n")
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", "s")
    b.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
    b.build()
    b.build_temporal("timestamp", "1d")
    return path


@pytest.fixture(scope="module", params=["single_pass", "chunked", "standalone_rebuild"])
def built_sphere(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from hypertopos.builder import GDSBuilder
    from hypertopos.storage.writer import GDSWriter

    path = tmp_path_factory.mktemp(f"traj_active_{request.param}") / "sphere"
    if request.param == "single_pass":
        return _build(path)
    if request.param == "standalone_rebuild":
        _build(path)
        GDSWriter(str(path)).build_trajectory_index("acct_pattern")
        return path
    mp = pytest.MonkeyPatch()
    mp.setattr(GDSBuilder, "_compute_chunk_size", lambda self, n, *a, **k: 2)
    try:
        return _build(path)
    finally:
        mp.undo()


def _traj_row(sphere: Path, key: str) -> dict | None:
    tbl = lance.dataset(str(sphere / "_gds_meta" / "trajectory" / "acct_pattern.lance")).to_table()
    rows = tbl.filter(pc.equal(tbl["primary_key"], key)).to_pylist()
    return rows[0] if rows else None


def _active_window_recipe(sphere: Path, key: str) -> tuple[np.ndarray, int, datetime, float]:
    """π9's view of one entity: stored windows, inactive ones dropped, pattern-metric deltas."""
    from hypertopos.engine.geometry import slice_deltas
    from hypertopos.navigation.navigator import active_slice_mask
    from hypertopos.storage.reader import GDSReader

    tbl = lance.dataset(str(sphere / "temporal" / "acct_pattern" / "data.lance")).to_table(
        columns=["primary_key", "shape_snapshot", "timestamp"]
    )
    tbl = tbl.filter(pc.equal(tbl["primary_key"], key)).sort_by("timestamp")
    shapes = np.array(tbl["shape_snapshot"].to_pylist(), dtype=np.float32)
    active = active_slice_mask(shapes)
    pattern = GDSReader(str(sphere)).read_sphere().patterns["acct_pattern"]
    deltas = slice_deltas(shapes[active], pattern)
    vec = np.concatenate([deltas.mean(axis=0), deltas.std(axis=0)])
    first_ts = tbl["timestamp"].to_pylist()[int(np.flatnonzero(active)[0])]
    disp = float(np.linalg.norm(deltas[-1] - deltas[0]))
    return vec, int(active.sum()), first_ts, disp


def test_build_rule_matches_drift_rule() -> None:
    from hypertopos.engine.geometry import active_window_rows
    from hypertopos.navigation.navigator import active_slice_mask

    rng = np.random.default_rng(3)
    random_shapes = rng.integers(0, 3, size=(12, 8)).astype(np.float32)
    random_shapes[:, 0] = 5.0  # a broadcast anchor-level dim
    cases = [
        np.array([[1.0, 2.0], [0.0, 2.0], [0.0, 2.0]]),
        np.ones((3, 2)),
        np.zeros((3, 2)),
        random_shapes,
    ]
    for shapes in cases:
        np.testing.assert_array_equal(
            active_window_rows(shapes, np.zeros(1, dtype=np.intp)), active_slice_mask(shapes)
        )
    # Grouped form: the same rule per entity.
    groups = cases[:3]
    starts = np.cumsum([0] + [len(g) for g in groups[:-1]])
    np.testing.assert_array_equal(
        active_window_rows(np.concatenate(groups), starts),
        np.concatenate([active_slice_mask(g) for g in groups]),
    )


def test_late_starter_vector_uses_only_active_windows(built_sphere: Path) -> None:
    row = _traj_row(built_sphere, _LATE)
    assert row is not None
    vec, n_active, first_ts, disp = _active_window_recipe(built_sphere, _LATE)
    assert n_active == 2
    assert row["num_slices"] == n_active
    assert row["first_timestamp"] == first_ts
    assert row["first_timestamp"] == datetime(2024, 1, 4, 12, tzinfo=UTC)
    np.testing.assert_allclose(row["trajectory_vector"], vec, rtol=1e-4, atol=1e-5)
    assert row["displacement"] == pytest.approx(disp, rel=1e-4, abs=1e-5)


def test_busy_entity_vector_matches_active_window_recipe(built_sphere: Path) -> None:
    row = _traj_row(built_sphere, _BUSY[0])
    assert row is not None
    vec, n_active, _first_ts, _disp = _active_window_recipe(built_sphere, _BUSY[0])
    assert row["num_slices"] == n_active
    np.testing.assert_allclose(row["trajectory_vector"], vec, rtol=1e-4, atol=1e-5)


def test_single_active_window_entity_has_no_trajectory(built_sphere: Path) -> None:
    # π9 needs two active windows to measure drift; π10 keeps the same population.
    assert _traj_row(built_sphere, _ONCE) is None


def test_drift_and_trajectory_agree_on_windows(built_sphere: Path) -> None:
    from hypertopos.sphere import HyperSphere

    session = HyperSphere.open(str(built_sphere)).session(agent_id="traj_active")
    try:
        nav = session.navigator()
        drift = {r["primary_key"]: r for r in nav.π9_attract_drift("acct_pattern", top_n=50)}
        with pytest.raises(ValueError, match="insufficient_temporal_history"):
            nav.π10_attract_trajectory(_ONCE, "acct_pattern")
    finally:
        session.close()
    assert _ONCE not in drift
    late = _traj_row(built_sphere, _LATE)
    assert late is not None
    assert drift[_LATE]["num_slices"] == late["num_slices"]
    assert drift[_LATE]["first_timestamp"] == late["first_timestamp"].isoformat()
