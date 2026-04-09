# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from hypertopos.storage._schemas import EDGE_STRUCT_TYPE
from hypertopos.utils.arrow import delta_matrix_from_arrow, flatten_edges_for_sql


class TestDeltaMatrixFromArrow:
    def test_fixed_size_list_zero_copy(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        col = pa.FixedSizeListArray.from_arrays(
            pa.array([v for row in data for v in row], type=pa.float32()), 3
        )
        table = pa.table({"delta": col})
        result = delta_matrix_from_arrow(table)
        assert result.dtype == np.float32
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, np.array(data, dtype=np.float32))

    def test_variable_list_list_type(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        col = pa.array(data, type=pa.list_(pa.float32()))
        table = pa.table({"delta": col})
        result = delta_matrix_from_arrow(table)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, np.array(data, dtype=np.float32))

    def test_variable_list_raises_on_null_row(self):
        """ListType with null row must raise ValueError, not silently corrupt."""
        col = pa.array([[1.0, 2.0], None, [3.0, 4.0]], type=pa.list_(pa.float32()))
        table = pa.table({"delta": col})
        with pytest.raises(ValueError, match="null rows"):
            delta_matrix_from_arrow(table)

    def test_empty_table(self):
        col = pa.FixedSizeListArray.from_arrays(pa.array([], type=pa.float32()), 3)
        table = pa.table({"delta": col})
        result = delta_matrix_from_arrow(table)
        assert result.shape == (0, 3)
        assert result.dtype == np.float32

    def test_single_row(self):
        col = pa.FixedSizeListArray.from_arrays(pa.array([1.5, 2.5, 3.5], type=pa.float32()), 3)
        table = pa.table({"delta": col})
        result = delta_matrix_from_arrow(table)
        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result[0], [1.5, 2.5, 3.5])

    def test_large_dimension(self):
        n, d = 100, 20
        flat = np.random.randn(n * d).astype(np.float32)
        col = pa.FixedSizeListArray.from_arrays(pa.array(flat), d)
        table = pa.table({"delta": col})
        result = delta_matrix_from_arrow(table)
        assert result.shape == (n, d)
        np.testing.assert_array_almost_equal(result, flat.reshape(n, d))

    def test_variable_list_zero_copy_large(self):
        """ListType must use col.values path (not to_pylist) — verified by correctness at scale."""
        n, d = 50_000, 7
        data = np.random.rand(n, d).astype(np.float32)
        col = pa.array(data.tolist(), type=pa.list_(pa.float32()))
        table = pa.table({"delta": col})
        result = delta_matrix_from_arrow(table)
        assert result.shape == (n, d)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, data, decimal=5)

    def test_variable_list_empty(self):
        """Empty ListType returns (0, 0) array."""
        col = pa.array([], type=pa.list_(pa.float32()))
        table = pa.table({"delta": col})
        result = delta_matrix_from_arrow(table)
        assert result.shape == (0, 0)
        assert result.dtype == np.float32


def _make_geo_table(edges_rows: list[list[dict]]) -> pa.Table:
    return pa.table({
        "primary_key": pa.array([f"PK-{i}" for i in range(len(edges_rows))]),
        "edges": pa.array(edges_rows, type=pa.list_(EDGE_STRUCT_TYPE)),
    })


class TestFlattenEdgesForSql:
    def test_basic_alive_only(self):
        table = _make_geo_table([
            [
                {"line_id": "customers", "point_key": "C1", "status": "alive", "direction": "in"},
                {"line_id": "products", "point_key": "P1", "status": "dead", "direction": "out"},
                {"line_id": "regions", "point_key": "R1", "status": "alive", "direction": "in"},
            ],
        ])
        result = flatten_edges_for_sql(table)
        assert "edge_line_ids" in result.schema.names
        assert "edge_point_keys" in result.schema.names
        lids = result["edge_line_ids"][0].as_py()
        pkeys = result["edge_point_keys"][0].as_py()
        assert lids == ["customers", "regions"]
        assert pkeys == ["C1", "R1"]

    def test_multiple_rows(self):
        table = _make_geo_table([
            [
                {"line_id": "customers", "point_key": "C1", "status": "alive", "direction": "in"},
            ],
            [
                {"line_id": "customers", "point_key": "C2", "status": "alive", "direction": "in"},
                {"line_id": "products", "point_key": "P2", "status": "alive", "direction": "out"},
            ],
        ])
        result = flatten_edges_for_sql(table)
        assert result["edge_line_ids"][0].as_py() == ["customers"]
        assert result["edge_line_ids"][1].as_py() == ["customers", "products"]
        assert result["edge_point_keys"][0].as_py() == ["C1"]
        assert result["edge_point_keys"][1].as_py() == ["C2", "P2"]

    def test_all_dead_edges(self):
        table = _make_geo_table([
            [
                {"line_id": "customers", "point_key": "C1", "status": "dead", "direction": "in"},
            ],
        ])
        result = flatten_edges_for_sql(table)
        assert result["edge_line_ids"][0].as_py() == []
        assert result["edge_point_keys"][0].as_py() == []

    def test_empty_edges(self):
        table = _make_geo_table([[]])
        result = flatten_edges_for_sql(table)
        assert result["edge_line_ids"][0].as_py() == []
        assert result["edge_point_keys"][0].as_py() == []

    def test_preserves_original_columns(self):
        table = _make_geo_table([
            [
                {"line_id": "customers", "point_key": "C1", "status": "alive", "direction": "in"},
            ],
        ])
        result = flatten_edges_for_sql(table)
        assert "primary_key" in result.schema.names
        assert "edges" in result.schema.names
        assert result["primary_key"][0].as_py() == "PK-0"

    def test_result_equivalence_with_reference(self):
        """Vectorized result must match the reference Python-loop implementation."""
        edges_rows = [
            [
                {"line_id": "A", "point_key": "a1", "status": "alive", "direction": "in"},
                {"line_id": "B", "point_key": "b1", "status": "dead", "direction": "out"},
                {"line_id": "C", "point_key": "c1", "status": "alive", "direction": "in"},
            ],
            [
                {"line_id": "A", "point_key": "a2", "status": "dead", "direction": "in"},
                {"line_id": "B", "point_key": "b2", "status": "alive", "direction": "out"},
            ],
            [],
            [
                {"line_id": "A", "point_key": "a3", "status": "alive", "direction": "in"},
            ],
        ]
        table = _make_geo_table(edges_rows)

        # Reference: Python loop (the old implementation)
        ref_lids = []
        ref_pkeys = []
        for row_edges in edges_rows:
            lids = [e["line_id"] for e in row_edges if e["status"] == "alive"]
            pks = [e["point_key"] for e in row_edges if e["status"] == "alive"]
            ref_lids.append(lids)
            ref_pkeys.append(pks)

        result = flatten_edges_for_sql(table)
        for i in range(len(edges_rows)):
            assert result["edge_line_ids"][i].as_py() == ref_lids[i]
            assert result["edge_point_keys"][i].as_py() == ref_pkeys[i]
