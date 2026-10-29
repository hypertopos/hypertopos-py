# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""``score_per_group``: each group's calibrated score against its own reference."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from hypertopos.builder import builder as builder_module
from hypertopos.builder.builder import EdgeTableConfig, GDSBuilder, RelationSpec
from hypertopos.cli.schema import LineConfig, _parse_one_pattern
from hypertopos.engine import robust
from hypertopos.storage.exceptions import GDSStorageError
from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter

N_ACCOUNTS = 240
N_RARE = 10
N_TX = 4_000
DAY = 86_400.0
DIMS = ("tx_out_count", "sum_out", "tx_in_count")
PID = "account_pattern"


def _tables(seed: int = 0) -> tuple[pa.Table, pa.Table]:
    rng = np.random.default_rng(seed)
    segment = rng.choice(["retail", "corporate"], N_ACCOUNTS)
    segment[:N_RARE] = "rare"  # below the reference size: scored by the population
    accounts = pa.table(
        {
            "primary_key": [f"A{i:03d}" for i in range(N_ACCOUNTS)],
            "segment": segment.tolist(),
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


def _builder(
    out: Path,
    *,
    score_per_group: bool = True,
    axes: list[str] | None = None,
    score_min_group_size: int | None = None,
) -> GDSBuilder:
    accounts, tx = _tables()
    b = GDSBuilder("per_group_test", str(out))
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
        PID,
        "anchor",
        "accounts",
        relations=[],
        anomaly_percentile=95.0,
        group_by_property="segment",
        score_axes=axes,
        score_per_group=score_per_group,
        score_min_group_size=score_min_group_size,
    )
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, DIMS[0])
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", DIMS[1])
    b.add_derived_dimension("accounts", "transactions", "to_account", "count", None, DIMS[2])
    return b


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("per_group") / "sphere"
    _builder(out).build()
    return out


@pytest.fixture(scope="module")
def built_mahal(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("per_group_mahal") / "sphere"
    _builder(out, axes=["mahal"]).build()
    return out


def _geometry(out: Path) -> pa.Table:
    table = lance.dataset(str(out / "geometry" / PID / "data.lance")).to_table()
    return table.sort_by("primary_key")


def _segments(keys: pa.Array) -> np.ndarray:
    accounts, _ = _tables()
    idx = pc.index_in(keys, value_set=accounts["primary_key"])
    return np.asarray(pc.take(accounts["segment"], idx).to_pylist(), dtype=object)


def _natural_x(keys: pa.Array) -> np.ndarray:
    _, tx = _tables()
    cols = []
    for fk, metric in (("from_account", "count"), ("from_account", "sum"), ("to_account", "count")):
        grouped = tx.group_by(fk).aggregate([("amount", metric)])
        idx = pc.index_in(keys, value_set=grouped[fk])
        values = pc.take(grouped[f"amount_{metric}"], idx)
        cols.append(pc.fill_null(values, 0).to_numpy(zero_copy_only=False).astype(np.float64))
    return np.column_stack(cols)


def _left_rank_pct(values: np.ndarray) -> np.ndarray:
    return 100.0 * np.searchsorted(np.sort(values), values, side="left") / len(values)


# ── builder ────────────────────────────────────────────────────────────


def test_score_per_group_requires_group_by_property(tmp_path: Path) -> None:
    b = GDSBuilder("t", str(tmp_path / "s"))
    with pytest.raises(ValueError, match="score_per_group requires group_by_property"):
        b.add_pattern("p", "anchor", "accounts", relations=[], score_per_group=True)


def test_score_per_group_requires_a_calibrated_score(tmp_path: Path) -> None:
    accounts, _ = _tables()
    b = GDSBuilder("t", str(tmp_path / "s"))
    b.add_line("accounts", accounts, key_col="primary_key", source_id="t")
    b.add_pattern(
        "p",
        "anchor",
        "accounts",
        relations=[RelationSpec("accounts", fk_col=None, direction="self")],
        group_by_property="segment",
        score_per_group=True,
    )
    with pytest.raises(ValueError, match="score_per_group needs the calibrated evidence score"):
        b.build()


def test_score_group_column_holds_each_rows_reference_key(built: Path) -> None:
    geo = _geometry(built)
    segments = _segments(geo["primary_key"])
    assert set(segments) == {"retail", "corporate", "rare"}
    expected = [None if s == "rare" else s for s in segments]
    assert geo["score_group"].to_pylist() == expected


def test_flags_and_p_come_from_each_groups_reference(built: Path) -> None:
    reader = GDSReader(str(built))
    population = reader.read_reference(PID)
    geo = _geometry(built)
    groups = np.asarray(geo["score_group"].to_pylist(), dtype=object)
    p = geo["conformal_p"].to_numpy().astype(np.float64)
    score = geo["anomaly_score"].to_numpy()
    for key in ("retail", "corporate"):
        m = groups == key
        ref = reader.read_reference(PID, group=key)
        assert ref.n_fitted == int(m.sum())
        np.testing.assert_allclose(p[m], robust.conformal_p(ref.cal_sorted, score[m]), rtol=1e-6)
    rare = np.array([g is None for g in groups])
    np.testing.assert_allclose(
        p[rare], robust.conformal_p(population.cal_sorted, score[rare]), rtol=1e-6
    )
    assert population.n_fitted == N_ACCOUNTS
    np.testing.assert_array_equal(geo["is_anomaly"].to_numpy(), robust.is_anomaly(p, 95.0))


def test_scores_equal_fit_grouped_reference(built_mahal: Path) -> None:
    geo = _geometry(built_mahal)
    x = _natural_x(geo["primary_key"])
    _, expected, keys = robust.fit_grouped_reference(
        x, DIMS, _segments(geo["primary_key"]), percentile=95.0, axes=("mahal",)
    )
    assert geo["score_group"].to_pylist() == keys.tolist()
    np.testing.assert_allclose(
        geo["conformal_p"].to_numpy(), expected.conformal_p, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(geo["anomaly_score"].to_numpy(), expected.anomaly_score, rtol=1e-6)
    np.testing.assert_allclose(
        geo["score_rank_pct"].to_numpy(), expected.score_rank_pct, rtol=1e-5, atol=1e-4
    )
    np.testing.assert_array_equal(
        geo["is_anomaly"].to_numpy(), robust.is_anomaly(expected.conformal_p, 95.0)
    )


def test_per_group_scores_differ_from_the_population(built_mahal: Path) -> None:
    geo = _geometry(built_mahal)
    x = _natural_x(geo["primary_key"])
    _, population = robust.fit_reference(x, DIMS, axes=("mahal",))
    assert not np.allclose(geo["conformal_p"].to_numpy(), population.conformal_p, rtol=1e-5)


def test_sphere_json_marks_the_pattern_and_bumps_the_format(built: Path) -> None:
    meta = json.loads((built / "_gds_meta" / "sphere.json").read_text())
    assert meta["format_version"] == "4.2"
    assert meta["patterns"][PID]["score_per_group"] is True
    assert "score_per_group" not in meta["patterns"]["tx_pattern"]
    pattern = GDSReader(str(built)).read_sphere().patterns[PID]
    assert pattern.score_per_group is True


def test_default_grouped_pattern_is_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "default"
    _builder(out, score_per_group=False).build()
    meta = json.loads((out / "_gds_meta" / "sphere.json").read_text())
    assert meta["format_version"] == "4.0"
    assert "score_per_group" not in meta["patterns"][PID]
    geo = _geometry(out)
    assert "score_group" not in geo.column_names
    with np.load(out / "_gds_meta" / "reference" / PID / "v=1.npz") as data:
        assert "group_names" not in data.files
    with pytest.raises(GDSStorageError, match="no per-group references"):
        GDSReader(str(out)).read_reference(PID, group="retail")


def test_chunked_build_writes_score_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder_module, "GEOMETRY_CHUNK_SIZE", 100)
    out = tmp_path / "chunked"
    _builder(out).build()
    geo = _geometry(out)
    segments = _segments(geo["primary_key"])
    assert geo["score_group"].to_pylist() == [None if s == "rare" else s for s in segments]
    groups = np.asarray(geo["score_group"].to_pylist(), dtype=object)
    ref = GDSReader(str(out)).read_reference(PID, group="retail")
    m = groups == "retail"
    np.testing.assert_allclose(
        geo["conformal_p"].to_numpy()[m],
        robust.conformal_p(ref.cal_sorted, geo["anomaly_score"].to_numpy()[m]),
        rtol=1e-6,
    )


def test_yaml_key_is_parsed() -> None:
    lines = {"accounts": LineConfig(source="s", key="primary_key")}
    spec = {"type": "anchor", "entity_line": "accounts", "group_by_property": "segment"}
    assert _parse_one_pattern("p", spec, lines, set()).score_per_group is False
    cfg = _parse_one_pattern("p", {**spec, "score_per_group": True}, lines, set())
    assert cfg.score_per_group is True
    with pytest.raises(ValueError, match="score_per_group must be a boolean"):
        _parse_one_pattern("p", {**spec, "score_per_group": "yes"}, lines, set())


# ── storage ────────────────────────────────────────────────────────────


def test_read_reference_unknown_group_raises(built: Path) -> None:
    with pytest.raises(GDSStorageError, match="no score group 'rare'"):
        GDSReader(str(built)).read_reference(PID, group="rare")


def test_read_grouped_reference_round_trips(built: Path) -> None:
    reader = GDSReader(str(built))
    grouped = reader.read_grouped_reference(PID)
    assert set(grouped.groups) == {"retail", "corporate"}
    assert grouped.pooled == frozenset({"rare"})
    assert grouped.resolve("rare")[1:] == (None, "global")
    np.testing.assert_array_equal(
        grouped.groups["retail"].cal_sorted, reader.read_reference(PID, group="retail").cal_sorted
    )
    np.testing.assert_array_equal(
        grouped.population.cal_sorted, reader.read_reference(PID).cal_sorted
    )


def test_rank_recompute_keeps_the_within_group_ranks(tmp_path: Path) -> None:
    out = tmp_path / "ranks"
    b = _builder(out, axes=["mahal"])
    b.build()
    before = _geometry(out)
    ranks = before["score_rank_pct"].to_numpy()
    groups = np.asarray(before["score_group"].to_pylist(), dtype=object)
    m = groups == "retail"
    np.testing.assert_allclose(
        ranks[m], _left_rank_pct(before["anomaly_score"].to_numpy()[m]), atol=1e-4
    )
    # A whole-population rank of the stored scores would differ.
    assert not np.allclose(ranks, _left_rank_pct(before["anomaly_score"].to_numpy()), atol=1e-4)
    b.finalize_incremental(PID)
    GDSWriter(str(out)).recompute_rank_stats(PID)
    np.testing.assert_array_equal(_geometry(out)["score_rank_pct"].to_numpy(), ranks)


def test_incremental_update_refuses_a_per_group_pattern(tmp_path: Path) -> None:
    out = tmp_path / "incr"
    b = _builder(out)
    b.build()
    accounts, _ = _tables()
    changed = accounts.slice(0, 5)
    with pytest.raises(ValueError, match="group_by_property"):
        b.incremental_update(PID, changed_entities=changed)
