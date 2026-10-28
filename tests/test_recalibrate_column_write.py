# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Recalibrate replaces only the calibration-dependent geometry columns."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pytest

from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.writer import GDSWriter
from tests.conftest import FIXTURES_PATH, clone_sphere

PID = "customer_pattern"


@pytest.fixture
def sphere_dir(tmp_path: Path) -> Path:
    base = clone_sphere(FIXTURES_PATH, tmp_path / "s")
    GDSWriter(str(base)).build_index_if_needed(PID)
    return base


def _geo(base: Path) -> Path:
    return base / "geometry" / PID / "data.lance"


def _uuids(ds: lance.LanceDataset) -> dict[str, list[str]]:
    return {i.name: sorted(str(s.uuid) for s in i.segments) for i in ds.describe_indices()}


def _check_columns(ds: lance.LanceDataset, theta_norm: float) -> None:
    t = ds.to_table()
    delta = np.stack(t["delta"].to_numpy(zero_copy_only=False))
    norms = t["delta_norm"].to_numpy()
    np.testing.assert_allclose(norms, np.sqrt((delta.astype(np.float64) ** 2).sum(1)), rtol=1e-5)
    assert t["is_anomaly"].to_pylist() == (norms >= theta_norm).tolist()
    for k in range(delta.shape[1]):
        np.testing.assert_array_equal(t[f"delta_dim_{k}"].to_numpy(), delta[:, k])


def test_main_recalibrate_keeps_the_primary_key_index(sphere_dir: Path) -> None:
    before = _uuids(lance.dataset(str(_geo(sphere_dir))))
    assert "primary_key_idx" in before

    HyperSphere.open(str(sphere_dir)).session("t").recalibrate(PID)

    ds = lance.dataset(str(_geo(sphere_dir)))
    after = _uuids(ds)
    assert after["primary_key_idx"] == before["primary_key_idx"]
    for name in ("delta_norm_idx", "score_rank_pct_idx", "is_anomaly_idx"):
        assert ds.stats.index_stats(name)["num_unindexed_rows"] == 0, name
    theta = json.loads((sphere_dir / "_gds_meta" / "sphere.json").read_text("utf-8"))
    theta_norm = float(np.linalg.norm(theta["patterns"][PID]["theta"]))
    _check_columns(ds, theta_norm)


def test_branch_recalibrate_writes_columns_on_the_branch_only(sphere_dir: Path) -> None:
    main_before = lance.dataset(str(_geo(sphere_dir))).version
    hs = HyperSphere.open(str(sphere_dir))
    hs.session("t").recalibrate(PID, branch="exp")

    main = lance.dataset(str(_geo(sphere_dir)))
    assert main.version == main_before
    head = lb.open_ref(_geo(sphere_dir), "exp")
    assert _uuids(head)["primary_key_idx"] == _uuids(main)["primary_key_idx"]
    recipe = lb.read_branch_json(sphere_dir, "exp")["recipes"][-1]
    assert recipe["lance_version_after"][PID] == head.version
    overlay = lb.overlay_dir(sphere_dir, "exp") / "_gds_meta" / "sphere.json"
    theta = json.loads(overlay.read_text("utf-8"))["patterns"][PID]["theta"]
    _check_columns(head, float(np.linalg.norm(theta)))
