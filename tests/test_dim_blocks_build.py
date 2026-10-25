# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Dimension blocks (metric_properties / geo_properties) on anchor patterns."""

from __future__ import annotations

import lance
import numpy as np
import pyarrow as pa
import pytest

import hypertopos.builder.builder as builder_module
from hypertopos.builder import GDSBuilder
from hypertopos.storage.reader import GDSReader

N = 3000


def _table() -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "k": [f"E{i}" for i in range(N)],
            "x": rng.normal(0, 5, N),
            "y": rng.lognormal(0, 1, N),
            "share": rng.uniform(0, 0.2, N),
        }
    )


def _build(path, *, precomputed: bool = True) -> None:
    b = GDSBuilder("blocks", str(path))
    b.add_line("e", _table(), key_col="k", source_id="t")
    b.add_pattern("p", "anchor", "e", relations=[], metric_properties=["x", "share"])
    if precomputed:
        b.add_precomputed_dimension("e", "y")
    b.build()


def _norms(path) -> dict[str, float]:
    t = lance.dataset(str(path / "geometry" / "p" / "data.lance")).to_table(
        columns=["primary_key", "delta_norm"]
    )
    return dict(zip(t["primary_key"].to_pylist(), t["delta_norm"].to_pylist(), strict=True))


def test_dim_labels_and_width_include_metric_dims(tmp_path):
    _build(tmp_path / "s")
    pattern = GDSReader(str(tmp_path / "s")).read_sphere().patterns["p"]
    assert pattern.dim_labels == ["_d_y", "t:x", "t:share"]
    assert pattern.delta_dim() == 3
    assert len(pattern.dimension_kinds or []) == 3


def test_large_pattern_keeps_metric_dims(tmp_path, monkeypatch):
    # Above GEOMETRY_CHUNK_SIZE the build writes in chunks; the metric block must
    # survive that path with the same geometry as the in-memory one.
    _build(tmp_path / "small")
    monkeypatch.setattr(builder_module, "GEOMETRY_CHUNK_SIZE", 1000)
    _build(tmp_path / "large")
    large = GDSReader(str(tmp_path / "large")).read_sphere().patterns["p"]
    assert len(large.mu) == 3
    small_norms = _norms(tmp_path / "small")
    for key, norm in _norms(tmp_path / "large").items():
        assert norm == pytest.approx(small_norms[key], rel=1e-5)


@pytest.mark.parametrize("precomputed", [False, True], ids=["blocks-only", "with-relation"])
def test_explain_anomaly_reads_the_stored_geometry(tmp_path, precomputed):
    # A polygon of a pattern with dimension blocks cannot be rebuilt from edges
    # (the blocks are not edges): the stored delta is the geometry.
    from hypertopos.sphere import HyperSphere

    _build(tmp_path / "s", precomputed=precomputed)
    session = HyperSphere.open(str(tmp_path / "s")).session("t")
    nav = session.navigator()
    top = max(_norms(tmp_path / "s").items(), key=lambda kv: kv[1])
    report = nav.explain_anomaly(top[0], "p")
    assert report is not None
    polygon = nav._engine.build_polygon(top[0], "p", nav._manifest)
    assert polygon.delta_norm == pytest.approx(top[1], rel=1e-6)


def test_pattern_with_only_metric_dims_builds(tmp_path):
    _build(tmp_path / "s", precomputed=False)
    pattern = GDSReader(str(tmp_path / "s")).read_sphere().patterns["p"]
    assert pattern.dim_labels == ["t:x", "t:share"]
    norms = np.array(list(_norms(tmp_path / "s").values()))
    x = _table()
    z = np.column_stack(
        [
            (x[c].to_numpy() - x[c].to_numpy().mean()) / max(x[c].to_numpy().std(), 1e-2)
            for c in ("x", "share")
        ]
    )
    np.testing.assert_allclose(np.sort(norms), np.sort(np.linalg.norm(z, axis=1)), rtol=1e-4)
