from __future__ import annotations

import numpy as np
import pyarrow as pa

from hypertopos.scoring import FrozenCalibration

DIMS = ("a", "b", "c")


def _table(n: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    seg = np.where(np.arange(n) % 2 == 0, "x", "y")
    shift = np.where(seg == "x", 0.0, 5.0)
    cols = {d: pa.array(rng.normal(size=n) + shift) for d in DIMS}
    return pa.table({**cols, "seg": pa.array(seg)})


def test_row_evidence_matches_score():
    table = _table(2_000, 1)
    cal = FrozenCalibration.fit(table, DIMS)
    evidence, refs = cal.row_evidence(table, members=True)
    scores = cal.score(table, members=True)
    np.testing.assert_array_equal(evidence.conformal_p, scores.conformal_p)
    np.testing.assert_array_equal(evidence.delta, scores.score_delta)
    assert evidence.axis_ranks.shape == (table.num_rows, len(cal.reference.axes))
    assert all(r is cal.reference for r in refs)


def test_row_evidence_names_each_rows_group_reference():
    table = _table(4_000, 2)
    cal = FrozenCalibration.fit(table, DIMS, group_by="seg", score_per_group=True)
    evidence, refs = cal.row_evidence(table, members=True)
    np.testing.assert_array_equal(evidence.conformal_p, cal.score(table, members=True).conformal_p)
    assert cal.groups is not None
    seg = table["seg"].to_pylist()
    for i in (0, 1, 2, 3):
        assert refs[i] is cal.groups[seg[i]].reference
