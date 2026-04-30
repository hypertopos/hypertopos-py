"""Builder integration: edge_dimensions on an event pattern produces sidecar + grown shape."""
from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.dataset as ds

from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.builder.builder import EdgeTableConfig
from hypertopos.builder.mapping import EdgeDimensionsConfig


def _make_event_sphere(out_root: Path, *, with_edge_dims: bool) -> GDSBuilder:
    b = GDSBuilder("test_s1", str(out_root))
    # Two anchor accounts.
    b.add_line(
        "accounts",
        [
            {"acct_id": "A", "name": "alpha"},
            {"acct_id": "B", "name": "beta"},
            {"acct_id": "C", "name": "gamma"},
            {"acct_id": "D", "name": "delta"},
        ],
        key_col="acct_id", source_id="t",
    )
    # Six transactions: a structuring chain A→B→C→D + two unrelated.
    b.add_line(
        "transactions",
        [
            {"tx_id": "ek1", "from_acct": "A", "to_acct": "B",
             "ts": 0.0,    "amount": 20000.0},
            {"tx_id": "ek2", "from_acct": "B", "to_acct": "C",
             "ts": 100.0,  "amount": 5000.0},
            {"tx_id": "ek3", "from_acct": "C", "to_acct": "D",
             "ts": 200.0,  "amount": 5000.0},
            {"tx_id": "ek4", "from_acct": "A", "to_acct": "B",
             "ts": 1000.0, "amount": 1000.0},
            {"tx_id": "ek5", "from_acct": "C", "to_acct": "D",
             "ts": 5000.0, "amount": 2000.0},
            {"tx_id": "ek6", "from_acct": "D", "to_acct": "A",
             "ts": 8000.0, "amount": 3000.0},
        ],
        key_col="tx_id", source_id="t",
    )

    edge_dims = (
        EdgeDimensionsConfig(dims={
            "pair_edge_count": {},
            "find_motif_structuring": {
                "time_window_hours": 1.0,
                "amt1_min": 10000.0,
                "amt2_max": 10000.0,
            },
        })
        if with_edge_dims else None
    )

    b.add_pattern(
        "tx_pattern",
        pattern_type="event",
        entity_line="transactions",
        relations=[
            RelationSpec(
                "accounts", fk_col="from_acct", direction="in", required=True,
            ),
        ],
        edge_table=EdgeTableConfig(
            from_col="from_acct", to_col="to_acct",
            timestamp_col="ts", amount_col="amount",
        ),
        edge_dimensions=edge_dims,
    )
    return b


def test_event_pattern_with_edge_dimensions_emits_sidecar(tmp_path: Path):
    out_root = tmp_path / "gds_test_s1"
    builder = _make_event_sphere(out_root, with_edge_dims=True)
    builder.build()

    sidecar = (
        out_root / "_gds_meta" / "edge_features" / "tx_pattern" / "data.lance"
    )
    assert sidecar.exists(), f"sidecar missing at {sidecar}"
    table = lance.dataset(str(sidecar)).to_table()
    assert "event_key" in table.column_names
    assert "pair_edge_count" in table.column_names
    assert "find_motif_structuring" in table.column_names
    # ek1, ek2, ek3 are part of structuring → flag=1; others 0.
    rows = {
        ek: ec
        for ek, ec in zip(
            table["event_key"].to_pylist(),
            table["find_motif_structuring"].to_pylist(),
            strict=False,
        )
    }
    assert rows["ek1"] == 1.0
    assert rows["ek2"] == 1.0
    assert rows["ek3"] == 1.0
    assert rows["ek4"] == 0.0
    assert rows["ek5"] == 0.0
    assert rows["ek6"] == 0.0


def test_event_pattern_polygon_shape_grows_by_edge_dim_count(tmp_path: Path):
    out_baseline = tmp_path / "gds_baseline"
    builder_b = _make_event_sphere(out_baseline, with_edge_dims=False)
    builder_b.build()
    pat_baseline = json.loads(
        (out_baseline / "_gds_meta" / "calibration" / "tx_pattern.json").read_text(
            encoding="utf-8",
        ),
    )
    D_baseline = len(pat_baseline["calibrated_mu"])

    out_with = tmp_path / "gds_with"
    builder_w = _make_event_sphere(out_with, with_edge_dims=True)
    builder_w.build()
    pat_with = json.loads(
        (out_with / "_gds_meta" / "calibration" / "tx_pattern.json").read_text(
            encoding="utf-8",
        ),
    )
    D_with = len(pat_with["calibrated_mu"])

    # 2 edge_dimensions declared → +2 dims.
    assert D_with == D_baseline + 2
    # And dimension_kinds reflects the new dims (last two = poisson + bernoulli).
    kinds = pat_with.get("dimension_kinds", [])
    if kinds:
        assert kinds[-2:] == ["poisson", "bernoulli"]


def test_event_pattern_no_edge_dimensions_no_sidecar(tmp_path: Path):
    out_root = tmp_path / "gds_no_ed"
    builder = _make_event_sphere(out_root, with_edge_dims=False)
    builder.build()
    sidecar = (
        out_root / "_gds_meta" / "edge_features" / "tx_pattern" / "data.lance"
    )
    assert not sidecar.exists()
