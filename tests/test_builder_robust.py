# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""The builder scores continuous anchor patterns with the calibrated evidence score."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from hypertopos.builder.builder import EdgeTableConfig, GDSBuilder, RelationSpec
from hypertopos.engine import robust
from hypertopos.storage.reader import GDSReader

N_ACCOUNTS = 240
N_TX = 4_000
DAY = 86_400.0
DIMS = ("tx_out_count", "sum_out", "tx_in_count")


def _tables(seed: int = 0) -> tuple[pa.Table, pa.Table]:
    rng = np.random.default_rng(seed)
    accounts = pa.table(
        {
            "primary_key": [f"A{i:03d}" for i in range(N_ACCOUNTS)],
            "segment": rng.choice(["retail", "corporate"], N_ACCOUNTS).tolist(),
        }
    )
    src = rng.integers(0, N_ACCOUNTS, N_TX)
    dst = (src + rng.integers(1, N_ACCOUNTS, N_TX)) % N_ACCOUNTS
    ts = 1.7e9 + rng.uniform(0, 60 * DAY, N_TX)
    tx = pa.table(
        {
            "primary_key": [f"T{i:05d}" for i in range(N_TX)],
            "from_account": [f"A{i:03d}" for i in src],
            "to_account": [f"A{i:03d}" for i in dst],
            "timestamp": pa.array((ts * 1e6).astype(np.int64), type=pa.timestamp("us", tz="UTC")),
            "amount": rng.lognormal(5.0, 1.0, N_TX),
        }
    )
    return accounts, tx


def _builder(out: Path, *, group_by: str | None = None) -> GDSBuilder:
    accounts, tx = _tables()
    b = GDSBuilder("robust_test", str(out))
    b.add_line("accounts", accounts, key_col="primary_key", source_id="t")
    b.add_line("transactions", tx, key_col="primary_key", source_id="t", role="event")
    b.add_pattern(
        "tx_pattern",
        "event",
        "transactions",
        relations=[
            RelationSpec("accounts", fk_col="from_account", direction="in", required=True),
            RelationSpec("accounts", fk_col="to_account", direction="in", required=False),
        ],
        edge_table=EdgeTableConfig(
            from_col="from_account",
            to_col="to_account",
            timestamp_col="timestamp",
            amount_col="amount",
        ),
    )
    b.add_pattern(
        "account_pattern",
        "anchor",
        "accounts",
        relations=[],
        anomaly_percentile=95.0,
        group_by_property=group_by,
    )
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, DIMS[0])
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", DIMS[1])
    b.add_derived_dimension("accounts", "transactions", "to_account", "count", None, DIMS[2])
    return b


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("robust") / "sphere"
    _builder(out).build()
    return out


def _geometry(out: Path) -> pa.Table:
    table = lance.dataset(str(out / "geometry" / "account_pattern" / "data.lance")).to_table()
    return table.sort_by("primary_key")


def _natural_x(out: Path, keys: pa.Array) -> np.ndarray:
    accounts, tx = _tables()
    cols = []
    for fk, metric in (("from_account", "count"), ("from_account", "sum"), ("to_account", "count")):
        grouped = tx.group_by(fk).aggregate([("amount", metric)])
        idx = pc.index_in(keys, value_set=grouped[fk])
        values = pc.take(grouped[f"amount_{metric}"], idx)
        cols.append(pc.fill_null(values, 0).to_numpy(zero_copy_only=False).astype(np.float64))
    return np.column_stack(cols)


def test_build_writes_reference_and_scores(built: Path) -> None:
    meta = json.loads((built / "_gds_meta" / "sphere.json").read_text())["patterns"]
    pat = meta["account_pattern"]
    assert pat["score_axes"] == ["mahal", "current"]
    assert pat["score_dims"] == list(DIMS)
    assert pat["reference_epoch"] == 1
    assert (built / "_gds_meta" / "reference" / "account_pattern" / "v=1.npz").exists()
    assert "dim_normality_pvalues" not in pat
    ref = GDSReader(str(built)).read_reference("account_pattern")
    assert ref.axes == ("mahal", "current") and ref.n_fitted == N_ACCOUNTS
    geo = _geometry(built)
    assert geo.num_rows == N_ACCOUNTS
    for col in ("anomaly_score", "axis_ranks", "conformal_p", "score_rank_pct", "score_delta"):
        assert col in geo.column_names
    assert "delta_rank_pct" not in geo.column_names
    assert geo["anomaly_confidence"].null_count == N_ACCOUNTS
    score = geo["anomaly_score"].to_numpy()
    np.testing.assert_allclose(
        geo["conformal_p"].to_numpy(), robust.conformal_p(ref.cal_sorted, score), rtol=1e-6
    )
    ranks = np.stack(geo["axis_ranks"].to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(ranks.mean(axis=1), score, atol=1e-6)


def test_score_delta_is_named_rank_gauss(built: Path) -> None:
    ref = GDSReader(str(built)).read_reference("account_pattern")
    assert ref.dims == DIMS
    pattern = GDSReader(str(built)).read_sphere().patterns["account_pattern"]
    assert pattern.score_dims == DIMS
    geo = _geometry(built)
    x = _natural_x(built, geo["primary_key"])
    z = robust.to_gauss(ref.cdf_grid, ref.cdf_levels, x)
    score_delta = np.stack(geo["score_delta"].to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(score_delta, z - ref.lw_location, atol=1e-5)


def test_delta_keeps_its_shape_space_meaning(built: Path) -> None:
    pattern = GDSReader(str(built)).read_sphere().patterns["account_pattern"]
    geo = _geometry(built)
    delta = np.stack(geo["delta"].to_numpy(zero_copy_only=False))
    norms = np.linalg.norm(delta, axis=1)
    np.testing.assert_allclose(geo["delta_norm"].to_numpy(), norms, rtol=1e-5)
    assert delta.shape[1] == len(pattern.mu)


def test_is_anomaly_matches_percentile(built: Path) -> None:
    geo = _geometry(built)
    p = geo["conformal_p"].to_numpy().astype(np.float64)
    expected = robust.is_anomaly(p, 95.0)
    assert expected.any()
    np.testing.assert_array_equal(geo["is_anomaly"].to_numpy(), expected)


def test_dim_percentiles_come_from_the_sketch(built: Path) -> None:
    pat = json.loads((built / "_gds_meta" / "sphere.json").read_text())["patterns"][
        "account_pattern"
    ]
    ref = GDSReader(str(built)).read_reference("account_pattern")
    x = _natural_x(built, _geometry(built)["primary_key"])
    for d, name in enumerate(DIMS):
        entry = pat["dim_percentiles"][name]
        assert entry["min"] == pytest.approx(round(float(x[:, d].min()), 4))
        assert entry["max"] == pytest.approx(round(float(x[:, d].max()), 4))
        p50 = float(np.interp(0.5, ref.cdf_levels, ref.cdf_grid[d]))
        assert entry["p50"] == pytest.approx(round(p50, 4))


def test_grouped_pattern_is_scored_globally(tmp_path: Path) -> None:
    out = tmp_path / "grouped"
    _builder(out, group_by="segment").build()
    pat = json.loads((out / "_gds_meta" / "sphere.json").read_text())["patterns"]["account_pattern"]
    assert pat["score_axes"] == ["mahal", "current"]
    assert "heteroscedasticity_diagnostic" not in pat
    geo = _geometry(out)
    expected = robust.is_anomaly(geo["conformal_p"].to_numpy().astype(np.float64), 95.0)
    np.testing.assert_array_equal(geo["is_anomaly"].to_numpy(), expected)


def test_event_pattern_keeps_delta_norm_score(built: Path) -> None:
    pat = json.loads((built / "_gds_meta" / "sphere.json").read_text())["patterns"]["tx_pattern"]
    assert "score_axes" not in pat and "reference_epoch" not in pat and "score_dims" not in pat
    geo = lance.dataset(str(built / "geometry" / "tx_pattern" / "data.lance")).to_table(
        columns=["delta_norm", "anomaly_score", "axis_ranks", "score_delta"]
    )
    np.testing.assert_allclose(
        geo["anomaly_score"].to_numpy(), geo["delta_norm"].to_numpy(), rtol=1e-6
    )
    assert geo["axis_ranks"].null_count == geo.num_rows
    assert geo["score_delta"].null_count == geo.num_rows


def test_label_graph_features_are_not_score_dims(tmp_path: Path) -> None:
    b = _builder(tmp_path / "graph")
    b.add_graph_features(
        "accounts",
        "transactions",
        "from_account",
        "to_account",
        ["out_degree", "connected_component"],
    )
    b.build()
    pat = json.loads((tmp_path / "graph" / "_gds_meta" / "sphere.json").read_text())["patterns"][
        "account_pattern"
    ]
    assert "out_degree" in pat["score_dims"]
    assert "connected_component" not in pat["score_dims"]


def test_composite_line_pattern_keeps_delta_norm_score(tmp_path: Path) -> None:
    out = tmp_path / "pairs"
    b = _builder(out)
    b.add_composite_line("account_pairs", "transactions", ["from_account", "to_account"])
    b.add_pattern("pair_pattern", "anchor", "account_pairs", relations=[])
    b.add_derived_dimension(
        "account_pairs", "transactions", ["from_account", "to_account"], "count", None, "pair_n"
    )
    b.build()
    meta = json.loads((out / "_gds_meta" / "sphere.json").read_text())["patterns"]
    assert "score_dims" not in meta["pair_pattern"]
    assert meta["account_pattern"]["score_dims"] == list(DIMS)
    pattern = GDSReader(str(out)).read_sphere().patterns["pair_pattern"]
    assert pattern.score_dims is None


def test_chain_pattern_is_not_a_score_candidate(tmp_path: Path) -> None:
    from hypertopos.builder.builder import _PatternReg

    b = _builder(tmp_path / "chain")
    b._chain_lines.add("tx_chains")
    chain = _PatternReg(
        pattern_id="chain_pattern",
        pattern_type="anchor",
        entity_line="tx_chains",
        relations=[
            RelationSpec("_d_chain_hop_count", fk_col="hop_count", direction="out", edge_max=5)
        ],
        anomaly_percentile=95.0,
    )
    assert not b._score_candidate(chain)
    on_regular_line = _PatternReg(
        pattern_id="p",
        pattern_type="anchor",
        entity_line="accounts",
        relations=chain.relations,
        anomaly_percentile=95.0,
    )
    assert not b._score_candidate(on_regular_line)
