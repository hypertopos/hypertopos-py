"""Navigation on a pattern with a reference per group: each entity is read against its group's."""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pytest
from test_builder_robust import DIMS, _tables

from hypertopos.builder.builder import EdgeTableConfig, GDSBuilder, RelationSpec
from hypertopos.engine import robust
from hypertopos.sphere import HyperSphere
from hypertopos.storage import _lance_branches as lb
from hypertopos.storage.reader import GDSReader

PID = "account_pattern"


def _build(out: Path) -> Path:
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
        score_per_group=True,
    )
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, DIMS[0])
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", DIMS[1])
    b.add_derived_dimension("accounts", "transactions", "to_account", "count", None, DIMS[2])
    b.build()
    return out


@pytest.fixture(scope="module")
def sphere(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build(tmp_path_factory.mktemp("per_group_nav") / "sphere")


def _geometry(sphere: Path) -> dict[str, dict]:
    table = lance.dataset(str(sphere / "geometry" / PID / "data.lance")).to_table(
        columns=["primary_key", "conformal_p", "axis_ranks", "score_delta", "score_group"]
    )
    return {row["primary_key"]: row for row in table.to_pylist()}


def _navigator(sphere: Path):  # type: ignore[no-untyped-def]
    return HyperSphere.open(str(sphere)).session(agent_id="per_group_nav").navigator()


def _top_key(geo: dict[str, dict], group: str) -> str:
    rows = [r for r in geo.values() if r["score_group"] == group]
    return min(rows, key=lambda r: r["conformal_p"])["primary_key"]


def test_explain_uses_the_entity_group_reference(sphere: Path) -> None:
    geo = _geometry(sphere)
    key = _top_key(geo, "corporate")
    row = geo[key]
    out = _navigator(sphere).explain_anomaly(key, PID)
    assert out["score_group"] == "corporate"
    ref = GDSReader(str(sphere)).read_reference(PID, group="corporate")
    expected = robust.attribute(
        ref,
        np.asarray(row["score_delta"], dtype=np.float64),
        np.asarray(row["axis_ranks"], dtype=np.float64),
        row["conformal_p"],
    )
    got = {d["dim"]: d["bits"] for d in out["top_dimensions"]}
    for dim, bits in got.items():
        assert bits == pytest.approx(expected.dim_bits[dim], abs=1e-3)


def test_simulating_the_current_value_keeps_the_entity_p(sphere: Path) -> None:
    geo = _geometry(sphere)
    key = _top_key(geo, "retail")
    nav = _navigator(sphere)
    probe = nav.simulate_dimension_change(
        key, pattern_id=PID, line_id="accounts", set_dimension={DIMS[0]: 0.0}
    )
    current = probe["dimensions_overridden"][0]["old_value"]
    out = nav.simulate_dimension_change(
        key, pattern_id=PID, line_id="accounts", set_dimension={DIMS[0]: float(current)}
    )
    assert out["conformal_p_after"] == pytest.approx(
        out["conformal_p_before"], rel=1e-6
    )  # stored as float32


def test_reliability_flags_use_the_group_reference(sphere: Path) -> None:
    geo = _geometry(sphere)
    reader = GDSReader(str(sphere))
    nav = _navigator(sphere)
    polygons = nav.π5_attract_anomaly(PID, top_n=10)[0]
    assert polygons
    for poly in polygons:
        flags = poly.reliability_flags  # type: ignore[attr-defined]
        group = geo[poly.primary_key]["score_group"]
        ref = reader.read_reference(PID, group=group)
        cov = np.linalg.pinv(ref.lw_precision, hermitian=True)
        shares = robust.dim_shares(ref, np.asarray(poly.score_delta, dtype=np.float64), cov)
        assert flags["dominant_dim_share"] == pytest.approx(float(shares.max()), abs=1e-4)


def test_alarm_rate_drift_is_silent_on_a_per_group_pattern(sphere: Path) -> None:
    assert _navigator(sphere)._check_alarm_rate_drift(PID) == []


def test_compare_branches_has_no_single_threshold(tmp_path: Path) -> None:
    base = _build(tmp_path / "s")
    lb.create_branch(base, "exp")
    report = _navigator(base).compare_branches(PID, "exp")
    assert report.theta_from is None and report.theta_to is None
    assert report.flipped_in == report.flipped_out == 0
    assert report.jaccard == 1.0
