"""find_anomalies' influence fields: one projected geometry read per pattern version."""

from __future__ import annotations

from pathlib import Path

from hypertopos.sphere import HyperSphere
from hypertopos.storage.reader import GDSReader

PID = "customer_pattern"


def test_influence_fields_read_two_columns_once_per_version(
    fixture_sphere_path: Path, monkeypatch
) -> None:
    calls: list[list[str] | None] = []
    real = GDSReader.read_geometry

    def counting(self, pattern_id, version, *args, **kwargs):
        if pattern_id == PID:
            calls.append(kwargs.get("columns"))
        return real(self, pattern_id, version, *args, **kwargs)

    session = HyperSphere.open(str(fixture_sphere_path)).session("influence")
    try:
        nav = session.navigator()
        keys = nav._storage.read_geometry(PID, nav._manifest.pattern_versions[PID])[
            "primary_key"
        ].to_pylist()
        monkeypatch.setattr(GDSReader, "read_geometry", counting)
        first = nav._attach_influence_fields_to_anomaly_entries(
            [{"primary_key": k} for k in keys], PID
        )
        second = nav._attach_influence_fields_to_anomaly_entries(
            [{"primary_key": k} for k in keys], PID
        )
    finally:
        session.close()

    assert calls == [["primary_key", "delta"]]
    assert first == second
    assert all(e["total_impact"] is not None for e in first)


def test_recalibrate_clears_the_influence_cache(tmp_path: Path) -> None:
    from tests.conftest import FIXTURES_PATH, clone_sphere

    base = clone_sphere(FIXTURES_PATH, tmp_path / "s")
    session = HyperSphere.open(str(base)).session("influence")
    try:
        nav = session.navigator()
        nav._attach_influence_fields_to_anomaly_entries([{"primary_key": "x"}], PID)
        assert nav._influence_cache
        session.recalibrate(PID)
        assert nav._influence_cache == {}
    finally:
        session.close()
