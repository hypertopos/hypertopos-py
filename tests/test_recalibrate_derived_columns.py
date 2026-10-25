"""recalibrate refreshes every geometry column that depends on the calibration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
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
    norms = t["delta_norm"].to_numpy().astype(np.float64)
    delta = np.stack(t["delta"].to_numpy(zero_copy_only=False))
    expected_p = np.array([(norms >= v).sum() / len(norms) for v in norms], dtype=np.float32)
    np.testing.assert_allclose(t["conformal_p"].to_numpy(), expected_p, rtol=1e-6)
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
