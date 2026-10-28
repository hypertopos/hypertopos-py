from __future__ import annotations

import lance
import numpy as np
import pyarrow as pa

from hypertopos.engine.lance_sql_agg import find_calibrated_anomalies
from hypertopos.navigation.navigator import _calibrated_order_value


def _geometry(tmp_path):
    table = pa.table(
        {
            "primary_key": ["a", "b", "c", "d", "e", "b"],
            "anomaly_score": [0.99, 0.97, 0.97, 0.40, 0.995, 0.10],
            "conformal_p": [0.002, 0.004, 0.004, 0.5, 0.001, 0.9],
            "is_anomaly": [True, True, True, False, True, False],
        }
    )
    path = str(tmp_path / "data.lance")
    lance.write_dataset(table, path)
    return path


def test_orders_anomalies_by_score_then_key(tmp_path):
    out = find_calibrated_anomalies(_geometry(tmp_path), top_n=10)
    assert out["keys"] == ["e", "a", "b", "c"]
    assert out["anomaly_scores"] == [0.995, 0.99, 0.97, 0.97]
    assert out["total_found"] == 4


def test_offset_pages_after_ordering(tmp_path):
    out = find_calibrated_anomalies(_geometry(tmp_path), top_n=2, offset=1)
    assert out["keys"] == ["a", "b"]


def test_non_anomalous_duplicate_does_not_leak(tmp_path):
    out = find_calibrated_anomalies(_geometry(tmp_path), top_n=10)
    assert out["keys"].count("b") == 1
    assert "d" not in out["keys"]


def test_p_value_orders_rows_scored_on_different_axes(tmp_path):
    # An ingested row is scored without the event axis: a lower anomaly_score
    # can carry the smaller p-value, and the p-value decides the order.
    table = pa.table(
        {
            "primary_key": ["built", "ingested"],
            "anomaly_score": [0.99, 0.97],
            "conformal_p": [0.004, 0.001],
            "is_anomaly": [True, True],
        }
    )
    path = str(tmp_path / "mixed.lance")
    lance.write_dataset(table, path)
    out = find_calibrated_anomalies(path, top_n=10)
    assert out["keys"] == ["ingested", "built"]


def test_in_process_order_value_matches_the_sql_order():
    p = np.array([0.004, 0.001, 0.004, 0.004])
    score = np.array([0.99, 0.97, 0.98, 0.99])
    value = _calibrated_order_value(p, score)
    # Descending value = p ascending, then score descending; exact ties share a value.
    assert list(np.argsort(-value, kind="stable")) == [1, 0, 3, 2]
    assert value[0] == value[3]
