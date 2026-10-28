# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Recalibration refits the score reference; incremental ingest scores against it."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from hypertopos.builder.builder import EdgeTableConfig, GDSBuilder, RelationSpec
from hypertopos.engine import robust
from hypertopos.sphere import HyperSphere
from hypertopos.storage.reader import GDSReader

N_ACCOUNTS = 200
N_TX = 3_000
DAY = 86_400.0
DIMS = ("tx_out_count", "sum_out", "tx_in_count")


def _build(out: Path) -> None:
    rng = np.random.default_rng(1)
    src = rng.integers(0, N_ACCOUNTS, N_TX)
    dst = (src + rng.integers(1, N_ACCOUNTS, N_TX)) % N_ACCOUNTS
    ts = 1.7e9 + rng.uniform(0, 40 * DAY, N_TX)
    b = GDSBuilder("recal_test", str(out))
    b.add_line(
        "accounts",
        pa.table({"primary_key": [f"A{i:03d}" for i in range(N_ACCOUNTS)]}),
        key_col="primary_key",
        source_id="t",
    )
    b.add_line(
        "transactions",
        pa.table(
            {
                "primary_key": [f"T{i:05d}" for i in range(N_TX)],
                "from_account": [f"A{i:03d}" for i in src],
                "to_account": [f"A{i:03d}" for i in dst],
                "timestamp": pa.array(
                    (ts * 1e6).astype(np.int64), type=pa.timestamp("us", tz="UTC")
                ),
                "amount": rng.lognormal(5.0, 1.0, N_TX),
            }
        ),
        key_col="primary_key",
        source_id="t",
        role="event",
    )
    b.add_pattern(
        "tx_pattern",
        "event",
        "transactions",
        relations=[
            RelationSpec("accounts", fk_col="from_account", direction="in", required=True),
            RelationSpec("accounts", fk_col="to_account", direction="in", required=False),
        ],
        edge_table=EdgeTableConfig("from_account", "to_account", "timestamp", "amount"),
    )
    b.add_pattern("account_pattern", "anchor", "accounts", relations=[])
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, DIMS[0])
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", DIMS[1])
    b.add_derived_dimension("accounts", "transactions", "to_account", "count", None, DIMS[2])
    b.build()


@pytest.fixture
def sphere(tmp_path: Path) -> Path:
    out = tmp_path / "sphere"
    _build(out)
    return out


def _geometry(out: Path) -> pa.Table:
    path = out / "geometry" / "account_pattern" / "data.lance"
    return lance.dataset(str(path)).to_table().sort_by("primary_key")


def _pattern_meta(out: Path) -> dict:
    return json.loads((out / "_gds_meta" / "sphere.json").read_text())["patterns"][
        "account_pattern"
    ]


def test_incremental_rows_scored_against_frozen_reference(sphere: Path) -> None:
    x_new = np.array([[3.0, 900.0, 2.0], [80.0, 1e7, 75.0]])
    new_rows = pa.table(
        {
            "primary_key": ["N1", "N2"],
            **{dim: pa.array(x_new[:, d]) for d, dim in enumerate(DIMS)},
        }
    )
    GDSBuilder("recal_test", str(sphere)).incremental_update(
        "account_pattern", changed_entities=new_rows, recalibrate="never"
    )
    assert _pattern_meta(sphere)["reference_epoch"] == 1
    ref = GDSReader(str(sphere)).read_reference("account_pattern")
    expected = robust.score(ref, x_new)
    geo = _geometry(sphere)
    rows = geo.filter(pa.compute.is_in(geo["primary_key"], pa.array(["N1", "N2"])))
    assert rows.num_rows == 2
    np.testing.assert_allclose(rows["anomaly_score"].to_numpy(), expected.anomaly_score)
    np.testing.assert_allclose(rows["conformal_p"].to_numpy(), expected.conformal_p, rtol=1e-6)
    np.testing.assert_array_equal(
        rows["is_anomaly"].to_numpy(), robust.is_anomaly(expected.conformal_p, 95.0)
    )
    assert _stats_anomalies(sphere) == int(geo["is_anomaly"].to_numpy().sum())
    score_delta = np.stack(rows["score_delta"].to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(score_delta, expected.delta, atol=1e-6)
    # The fitted rows keep their p-values (frozen reference, no population rewrite).
    old = geo.filter(pa.compute.invert(pa.compute.is_in(geo["primary_key"], rows["primary_key"])))
    np.testing.assert_allclose(
        old["conformal_p"].to_numpy(),
        robust.conformal_p(ref.cal_sorted, old["anomaly_score"].to_numpy()),
        rtol=1e-6,
    )


def _stats_anomalies(out: Path) -> int:
    path = out / "_gds_meta" / "geometry_stats" / "account_pattern_v1.json"
    return int(json.loads(path.read_text())["total_anomalies"])


def test_recalibrate_bumps_reference_epoch(sphere: Path) -> None:
    hs = HyperSphere.open(str(sphere))
    session = hs.session("recal_agent")
    try:
        session.recalibrate("account_pattern")
    finally:
        session.close()
    meta = _pattern_meta(sphere)
    assert meta["reference_epoch"] == 2
    assert meta["score_dims"] == list(DIMS)
    assert (sphere / "_gds_meta" / "reference" / "account_pattern" / "v=2.npz").exists()
    ref = GDSReader(str(sphere)).read_reference("account_pattern")
    geo = _geometry(sphere)
    assert geo.num_rows == N_ACCOUNTS
    np.testing.assert_allclose(
        geo["conformal_p"].to_numpy(),
        robust.conformal_p(ref.cal_sorted, geo["anomaly_score"].to_numpy()),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        geo["is_anomaly"].to_numpy(),
        robust.is_anomaly(geo["conformal_p"].to_numpy().astype(np.float64), 95.0),
    )
    assert _stats_anomalies(sphere) == int(geo["is_anomaly"].to_numpy().sum())
