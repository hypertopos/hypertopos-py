# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Recalibrating a ``group_by_property`` pattern per group, as a rebuild would."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from hypertopos.builder.builder import EdgeTableConfig, GDSBuilder, RelationSpec
from hypertopos.engine import robust
from hypertopos.engine.geometry import calibrated_deltas, calibrated_shapes
from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.reader import GDSReader
from tests.test_builder_score_per_group import PID, _builder, _tables

# Two retail and two corporate accounts, and two of the ten rare ones.
EXCLUDED = ["A010", "A011", "A012", "A013", "A000", "A001"]
_COLUMNS = ("delta", "delta_norm", "conformal_p", "is_anomaly", "anomaly_score", "n_anomalous_dims")


@pytest.fixture(scope="module", params=[False, True], ids=["global_score", "score_per_group"])
def built(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("grouped_recal") / "sphere"
    _builder(out, score_per_group=request.param).build()
    return out


def _geometry(out: Path, pid: str = PID, branch: str | None = None) -> pa.Table:
    ds = lance.dataset(str(out / "geometry" / pid / "data.lance"))
    if branch is not None:
        ds = ds.checkout_version((branch, None))
    return ds.to_table().sort_by("primary_key")


def _meta(out: Path, pid: str = PID) -> dict:
    return json.loads((out / "_gds_meta" / "sphere.json").read_text())["patterns"][pid]


def _column(table: pa.Table, name: str) -> np.ndarray:
    values = table[name].to_numpy(zero_copy_only=False)
    return np.stack(values) if values.dtype == object and name != "score_group" else values


def _assert_group_stats_equal(actual: dict, expected: dict) -> None:
    assert sorted(actual) == sorted(expected)
    for gid, stats in expected.items():
        for field in ("mu", "sigma_diag", "theta"):
            np.testing.assert_allclose(actual[gid][field], stats[field], rtol=1e-5, atol=1e-5)
        assert actual[gid]["population_size"] == stats["population_size"]


def _recalibrate(out: Path, pid: str = PID, **kwargs: object) -> dict:
    session = HyperSphere.open(str(out)).session("recal")
    try:
        return session.recalibrate(pid, **kwargs)
    finally:
        session.close()


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_recalibrate_reproduces_the_grouped_build(built: Path, tmp_path: Path) -> None:
    out = _copy(built, tmp_path / "s")
    before = _geometry(out)
    meta_before = _meta(out)
    _recalibrate(out)
    after = _geometry(out)
    rewritten = (*_COLUMNS, "score_rank_pct", "bregman_divergence", "axis_ranks", "score_delta")
    for name in rewritten:
        np.testing.assert_allclose(
            _column(after, name).astype(np.float64),
            _column(before, name).astype(np.float64),
            atol=1e-4,
            err_msg=name,
        )
    meta = _meta(out)
    _assert_group_stats_equal(meta["group_stats"], meta_before["group_stats"])
    for field in ("mu", "sigma_diag", "theta"):
        np.testing.assert_allclose(meta[field], meta_before[field], rtol=1e-5, err_msg=field)
    if meta_before.get("score_per_group"):
        assert after["score_group"].to_pylist() == before["score_group"].to_pylist()
        old = GDSReader(str(built)).read_grouped_reference(PID)
        new = GDSReader(str(out)).read_grouped_reference(PID)
        assert sorted(new.groups) == sorted(old.groups) and new.pooled == old.pooled
        assert meta["reference_epoch"] == meta_before["reference_epoch"] + 1


def test_recalibrated_history_records_the_group_stats(built: Path, tmp_path: Path) -> None:
    out = _copy(built, tmp_path / "s")
    _recalibrate(out, exclude_keys=EXCLUDED)
    fit = GDSReader(str(out)).read_calibration_fit(PID)
    _assert_group_stats_equal(fit.group_stats, _meta(out)["group_stats"])


def _segments() -> dict[str, str]:
    accounts, _ = _tables()
    return dict(
        zip(accounts["primary_key"].to_pylist(), accounts["segment"].to_pylist(), strict=True)
    )


def _rebuild_without(
    built: Path, out: Path, keys: list[str], monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A build of the same data without ``keys``, normalized like ``built``.

    The build caps each dimension at its p99; recalibration keeps the stored
    caps, so the rebuild pins them to compare the fit and not the scaling.
    """
    import tests.test_builder_score_per_group as base

    accounts, tx = _tables()
    keep = pc.invert(pc.is_in(accounts["primary_key"], pa.array(keys)))
    monkeypatch.setattr(base, "_tables", lambda seed=0: (accounts.filter(keep), tx))
    caps = {rel["display_name"]: rel["edge_max"] for rel in _meta(built)["relations"]}
    add = GDSBuilder.add_derived_dimension

    def pinned(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return add(self, *args, edge_max=caps[args[5]], **kwargs)

    monkeypatch.setattr(GDSBuilder, "add_derived_dimension", pinned)
    _builder(out, score_per_group=bool(_meta(built).get("score_per_group"))).build()
    return out


def test_exclude_keys_fit_equals_a_build_without_them(
    built: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    per_group = bool(_meta(built).get("score_per_group"))
    out = _copy(built, tmp_path / "s")
    _recalibrate(out, exclude_keys=EXCLUDED)
    rebuilt = _rebuild_without(built, tmp_path / "rebuilt", EXCLUDED, monkeypatch)

    _assert_group_stats_equal(_meta(out)["group_stats"], _meta(rebuilt)["group_stats"])
    after, expected = _geometry(out), _geometry(rebuilt)
    kept = pc.invert(pc.is_in(after["primary_key"], pa.array(EXCLUDED)))
    fitted = after.filter(kept)
    assert fitted["primary_key"].to_pylist() == expected["primary_key"].to_pylist()
    for name in _COLUMNS:
        np.testing.assert_allclose(
            _column(fitted, name).astype(np.float64),
            _column(expected, name).astype(np.float64),
            atol=1e-4,
            err_msg=name,
        )

    # The excluded rows keep their shapes and are measured against their
    # group's refitted statistics, and scored as new rows.
    rows = after.filter(pc.invert(kept))
    built_rows = _geometry(built).filter(pc.invert(kept))
    segments = _segments()
    old_stats, new_stats = _meta(built)["group_stats"], _meta(out)["group_stats"]
    for key, old_delta, new_delta in zip(
        rows["primary_key"].to_pylist(),
        _column(built_rows, "delta"),
        _column(rows, "delta"),
        strict=True,
    ):
        old, new = old_stats[segments[key]], new_stats[segments[key]]
        shape = calibrated_shapes(
            old_delta, mu=old["mu"], sigma_diag=old["sigma_diag"], dimension_weights=None
        )
        expected_delta = calibrated_deltas(
            shape, mu=new["mu"], sigma_diag=new["sigma_diag"], dimension_weights=None
        )
        np.testing.assert_allclose(new_delta, expected_delta[0], atol=1e-4)
    reader = GDSReader(str(out))
    score = rows["anomaly_score"].to_numpy()
    rank_pct = rows["score_rank_pct"].to_numpy()

    def _scored_against(ref: robust.Reference, i: int) -> None:
        # The stored rank is this row's score within that reference's fit.
        expected_pct = 100.0 * np.searchsorted(ref.cal_sorted, score[i], side="left")
        assert rank_pct[i] == pytest.approx(expected_pct / ref.n_fitted, abs=1e-3)

    if per_group:
        assert fitted["score_group"].to_pylist() == expected["score_group"].to_pylist()
        grouped = reader.read_grouped_reference(PID)
        reference = GDSReader(str(rebuilt)).read_grouped_reference(PID)
        assert sorted(grouped.groups) == sorted(reference.groups)
        assert grouped.pooled == reference.pooled
        for gid, ref in grouped.groups.items():
            assert ref.n_fitted == reference.groups[gid].n_fitted
        assert set(grouped.population.excluded_keys.tolist()) == set(EXCLUDED)
        for i, (key, group) in enumerate(
            zip(rows["primary_key"].to_pylist(), rows["score_group"].to_pylist(), strict=True)
        ):
            ref, expected_group, _ = grouped.resolve(segments[key])
            assert group == expected_group
            assert key in set(ref.excluded_keys.tolist())
            _scored_against(ref, i)
    else:
        ref = reader.read_reference(PID)
        assert set(ref.excluded_keys.tolist()) == set(EXCLUDED)
        for i in range(rows.num_rows):
            _scored_against(ref, i)


def test_exclude_keys_that_empty_a_group_raise(built: Path, tmp_path: Path) -> None:
    out = _copy(built, tmp_path / "s")
    rare = [key for key, segment in _segments().items() if segment == "rare"]
    with pytest.raises(ValueError, match="rare"):
        _recalibrate(out, exclude_keys=rare)


def test_branch_recalibrate_of_a_grouped_pattern(built: Path, tmp_path: Path) -> None:
    out = _copy(built, tmp_path / "s")
    before = _geometry(out)
    _recalibrate(out, branch="exp", exclude_keys=EXCLUDED)
    assert _geometry(out)["delta"].to_pylist() == before["delta"].to_pylist()
    main = _copy(built, tmp_path / "m")
    _recalibrate(main, exclude_keys=EXCLUDED)
    on_branch, on_main = _geometry(out, branch="exp"), _geometry(main)
    for name in (*_COLUMNS, "score_group") if "score_group" in on_main.column_names else _COLUMNS:
        a, b = _column(on_branch, name), _column(on_main, name)
        if name == "score_group":
            assert a.tolist() == b.tolist()
        else:
            np.testing.assert_allclose(a.astype(np.float64), b.astype(np.float64), atol=1e-6)


# ── a grouped pattern scored by delta_norm (per-group theta flags) ─────


EVENT_PID = "tx_pattern"
PROPS_PID = "account_props"


def _delta_norm_builder(out: Path, dimension_weights: str | None) -> GDSBuilder:
    """A grouped event pattern on amount, and a grouped anchor pattern on
    tracked properties; neither carries the calibrated score."""
    accounts, tx = _tables()
    rng = np.random.default_rng(3)
    tx = tx.append_column("channel", pa.array(rng.choice(["web", "branch", "atm"], tx.num_rows)))
    for name, rate in (("memo", 0.7), ("email", 0.4)):
        filled = np.where(rng.random(accounts.num_rows) < rate, "x", None)
        accounts = accounts.append_column(name, pa.array(filled.tolist(), type=pa.string()))
    b = GDSBuilder("grouped_delta_norm", str(out))
    b.add_line("accounts", accounts, key_col="primary_key", source_id="t")
    b.add_line("transactions", tx, key_col="primary_key", source_id="t", role="event")
    b.add_pattern(
        EVENT_PID,
        "event",
        "transactions",
        relations=[
            RelationSpec("accounts", fk_col="from_account", direction="in", required=True),
            RelationSpec("accounts", fk_col="to_account", direction="in", required=False),
        ],
        edge_table=EdgeTableConfig("from_account", "to_account", "timestamp", "amount"),
        group_by_property="channel",
        dimension_weights=dimension_weights,
    )
    b.add_event_dimension(EVENT_PID, "amount")
    b.add_pattern(
        PROPS_PID,
        "anchor",
        "accounts",
        relations=[],
        group_by_property="segment",
        tracked_properties=["memo", "email"],
        dimension_weights=dimension_weights,
    )
    return b


@pytest.mark.parametrize("dimension_weights", [None, "kurtosis"])
def test_recalibrate_reproduces_delta_norm_scored_grouped_builds(
    tmp_path: Path, dimension_weights: str | None
) -> None:
    out = tmp_path / "s"
    _delta_norm_builder(out, dimension_weights).build()
    assert _meta(out, PROPS_PID)["prop_columns"] == ["memo", "email"]
    for pid in (EVENT_PID, PROPS_PID):
        before = _geometry(out, pid)
        meta_before = _meta(out, pid)
        assert "reference_epoch" not in meta_before
        assert ("dimension_weights" in meta_before) == (dimension_weights is not None)
        _recalibrate(out, pid)
        after = _geometry(out, pid)
        for name in (
            "delta",
            "delta_norm",
            "conformal_p",
            "is_anomaly",
            "score_rank_pct",
            "bregman_divergence",
            "n_anomalous_dims",
        ):
            # Rows tied to float precision may swap one rank.
            n = after.num_rows
            atol = {"conformal_p": 1.5 / n, "score_rank_pct": 150.0 / n}.get(name, 1e-4)
            np.testing.assert_allclose(
                _column(after, name).astype(np.float64),
                _column(before, name).astype(np.float64),
                rtol=1e-5,
                atol=atol,
                err_msg=f"{pid}.{name}",
            )
        meta = _meta(out, pid)
        _assert_group_stats_equal(meta["group_stats"], meta_before["group_stats"])
        for field in ("mu", "sigma_diag", "theta"):
            np.testing.assert_allclose(meta[field], meta_before[field], rtol=1e-5, err_msg=field)


def test_recalibrate_keeps_a_custom_score_group_size(tmp_path: Path) -> None:
    out = tmp_path / "s"
    # Above both segment sizes: every row lands in the pool, which gets its own reference.
    _builder(out, score_min_group_size=200).build()
    assert _meta(out)["score_min_group_size"] == 200
    before = _geometry(out)["score_group"].to_pylist()
    assert set(before) == {robust.POOLED_GROUP}
    _recalibrate(out)
    assert _geometry(out)["score_group"].to_pylist() == before


def _gmm_sphere(out: Path) -> Path:
    accounts, tx = _tables()
    b = GDSBuilder("gmm_recal", str(out))
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
    b.add_pattern(PID, "anchor", "accounts", relations=[], gmm_n_components=2)
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "n_out")
    b.add_derived_dimension("accounts", "transactions", "to_account", "count", None, "n_in")
    b.build()
    return out


def test_recalibrate_refuses_a_gmm_pattern(tmp_path: Path) -> None:
    out = _gmm_sphere(tmp_path / "s")
    assert _meta(out)["gmm_components"]
    before = _geometry(out)
    with pytest.raises(ValueError, match="rebuild"):
        _recalibrate(out)
    with pytest.raises(ValueError, match="rebuild"):
        _recalibrate(out, branch="exp")
    assert _geometry(out).equals(before)
    assert "exp" not in lb.branch_names(out)


def test_incremental_update_refuses_a_gmm_pattern_before_writing(tmp_path: Path) -> None:
    out = _gmm_sphere(tmp_path / "s")
    before = _geometry(out)
    meta = _meta(out)
    accounts, _ = _tables()
    with pytest.raises(ValueError, match="gmm"):
        GDSBuilder("gmm_recal", str(out)).incremental_update(
            PID, changed_entities=accounts.slice(0, 5), recalibrate="never"
        )
    assert _geometry(out).equals(before)
    assert _meta(out)["computed_at"] == meta["computed_at"]


def test_deleting_from_a_gmm_pattern_refuses_a_recalibration_before_writing(
    tmp_path: Path,
) -> None:
    out = _gmm_sphere(tmp_path / "s")
    before = _geometry(out)
    key = before["primary_key"][0].as_py()
    builder = GDSBuilder("gmm_recal", str(out))
    with pytest.raises(ValueError, match="gmm"):
        builder.incremental_update(PID, deleted_keys=[key], recalibrate="force")
    assert _geometry(out).equals(before)
    builder.incremental_update(PID, deleted_keys=[key], recalibrate="never")
    assert _geometry(out).num_rows == before.num_rows - 1
