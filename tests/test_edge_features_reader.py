"""Storage reader API for the edge_features sidecar."""
from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa

from hypertopos.storage.reader import GDSReader
from hypertopos.storage._schemas import EDGE_FEATURES_SCHEMA


def _make_minimal_sphere(tmp: Path) -> Path:
    """Build a directory tree the GDSReader can open without crashing."""
    base = tmp / "s"
    (base / "_gds_meta").mkdir(parents=True)
    (base / "_gds_meta" / "sphere.json").write_text(
        '{"sphere_id": "s", "format_version": "3.0", '
        '"patterns": [], "lines": []}',
        encoding="utf-8",
    )
    return base


def test_read_edge_features_returns_empty_when_sidecar_absent(tmp_path: Path):
    base = _make_minimal_sphere(tmp_path)
    reader = GDSReader(base)
    table = reader.read_edge_features("tx_pattern")
    assert table.num_rows == 0
    assert set(table.schema.names) == set(EDGE_FEATURES_SCHEMA.names)


def test_read_edge_features_returns_data_when_sidecar_present(tmp_path: Path):
    base = _make_minimal_sphere(tmp_path)
    sidecar_dir = base / "_gds_meta" / "edge_features" / "tx_pattern"
    sidecar_dir.mkdir(parents=True)
    table = pa.table(
        {
            "event_key": ["ek1", "ek2"],
            "pair_edge_count":           pa.array([3.0, 1.0], type=pa.float32()),
            "position_in_chain":         pa.array([0.0, 5.0], type=pa.float32()),
            "time_since_pair_last_edge": pa.array([60.0, 999.0], type=pa.float32()),
            "pair_amount_zscore":        pa.array([0.5, 0.0], type=pa.float32()),
            "find_motif_structuring":    pa.array([1.0, 0.0], type=pa.float32()),
        },
        schema=EDGE_FEATURES_SCHEMA,
    )
    lance.write_dataset(table, str(sidecar_dir / "data.lance"), mode="overwrite")

    reader = GDSReader(base)
    out = reader.read_edge_features("tx_pattern")
    assert out.num_rows == 2
    assert out["event_key"].to_pylist() == ["ek1", "ek2"]
    assert out["pair_edge_count"].to_pylist() == [3.0, 1.0]
    assert out["find_motif_structuring"].to_pylist() == [1.0, 0.0]
