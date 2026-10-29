"""recalibrate refreshes every geometry column that depends on the calibration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

_ACCOUNTS = [f"A-{i:02d}" for i in range(30)]


@pytest.fixture
def built_sphere(tmp_path: Path) -> Path:
    from hypertopos.builder import GDSBuilder

    path = tmp_path / "sphere"
    rows = [
        {
            "tx_id": f"TX-{i}",
            "from_account": _ACCOUNTS[i % len(_ACCOUNTS)],
            "amount": float((i * 37) % 91 + 1) * (40.0 if i % 29 == 0 else 1.0),
            "timestamp": datetime(2024, 1, 1 + i % 5),
        }
        for i in range(240)
    ]
    b = GDSBuilder("test", str(path))
    b.add_line(
        "transactions",
        pa.table({k: [r[k] for r in rows] for k in rows[0]}),
        key_col="tx_id",
        source_id="test",
        role="event",
    )
    b.add_line(
        "accounts",
        pa.table({"primary_key": _ACCOUNTS}),
        key_col="primary_key",
        source_id="test",
        role="anchor",
    )
    b.add_derived_dimension("accounts", "transactions", "from_account", "count", None, "n")
    b.add_derived_dimension("accounts", "transactions", "from_account", "sum", "amount", "s")
    b.add_derived_dimension("accounts", "transactions", "from_account", "max", "amount", "m")
    b.add_pattern("acct_pattern", "anchor", "accounts", relations=[])
    b.build()
    return path


def test_recalibrate_refreshes_every_calibration_dependent_column(built_sphere: Path) -> None:
    from hypertopos.sphere import HyperSphere

    session = HyperSphere.open(str(built_sphere)).session("t")
    try:
        session.recalibrate("acct_pattern", exclude_keys=_ACCOUNTS[:12])
    finally:
        session.close()

    t = lance.dataset(str(built_sphere / "geometry" / "acct_pattern" / "data.lance")).to_table()
    delta = np.stack(t["delta"].to_numpy(zero_copy_only=False))
    # The pattern is on the calibrated evidence score: p is against the
    # refitted reference (fitted without the excluded keys).
    from hypertopos.engine.robust import conformal_p
    from hypertopos.storage.reader import GDSReader

    ref = GDSReader(str(built_sphere)).read_reference("acct_pattern")
    assert ref.n_fitted == len(_ACCOUNTS) - 12
    # Fitted rows carry their in-sample p; the excluded rows are scored as new
    # rows (out-of-fold calibration), so they are checked separately.
    fitted = ~np.isin(t["primary_key"].to_numpy(zero_copy_only=False), _ACCOUNTS[:12])
    expected_p = conformal_p(ref.cal_sorted, t["anomaly_score"].to_numpy())
    np.testing.assert_allclose(t["conformal_p"].to_numpy()[fitted], expected_p[fitted], rtol=1e-6)
    from hypertopos.engine.robust import counterfactual

    assert set(ref.excluded_keys.tolist()) == set(_ACCOUNTS[:12])
    for i in np.flatnonzero(~fitted):
        again = counterfactual(
            ref,
            np.asarray(t["score_delta"][int(i)].as_py()),
            np.asarray(t["axis_ranks"][int(i)].as_py()),
            {},
            members=False,
        )
        assert again.conformal_p == pytest.approx(t["conformal_p"][int(i)].as_py())

    # simulate_dimension_change on an excluded entity rescores it as the new row
    # it was: setting a score dimension to its own value leaves its p unchanged.
    assert ref.cal_sorted_oof is not None  # otherwise both paths agree trivially
    session = HyperSphere.open(str(built_sphere)).session("t2")
    try:
        pattern = session._reader.read_sphere().patterns["acct_pattern"]
        line = pattern.entity_line_id
        dim = ref.dims[0]
        points = session._reader.read_points(
            line, session._manifest.line_versions[line], columns=["primary_key", dim]
        )
        key = _ACCOUNTS[0]
        value = float(points.filter(pc.equal(points["primary_key"], key))[dim][0].as_py())
        out = session.navigator().simulate_dimension_change(
            key, pattern_id="acct_pattern", line_id=line, set_dimension={dim: value}
        )
    finally:
        session.close()
    stored = float(t.filter(pc.equal(t["primary_key"], key))["conformal_p"][0].as_py())
    assert out["conformal_p_after"] == pytest.approx(stored)
    # n_anomalous_dims counts against the per-dim thresholds stored on the
    # pattern node, the same rule as a full build.
    import json

    node = json.loads((built_sphere / "_gds_meta" / "sphere.json").read_text(encoding="utf-8"))[
        "patterns"
    ]["acct_pattern"]
    thresholds = np.asarray(node["n_anomalous_dims_thresholds"], dtype=np.float64)
    np.testing.assert_array_equal(
        t["n_anomalous_dims"].to_numpy(), (np.abs(delta) > thresholds).sum(axis=1)
    )
    assert t["anomaly_confidence"].null_count == t.num_rows
    assert t["delta_norm_signed"].null_count == t.num_rows
