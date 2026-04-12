# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for hypertopos.builder._scatter — vectorized Arrow→tensor scatter."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from hypertopos.builder._scatter import vectorized_scatter, vectorized_scatter_1d


class TestVectorizedScatter:
    def test_basic_scatter(self):
        anchor_keys = pa.array(["A", "B", "C"])
        grouped_fk = pa.array(["B", "A", "C", "A"])
        grouped_bucket = pa.array([0, 1, 0, 2], type=pa.int64())
        grouped_values = pa.array([10.0, 20.0, 30.0, 40.0])
        tensor = np.zeros((3, 4, 1), dtype=np.float32)

        vectorized_scatter(
            tensor, dim_idx=0, edge_max=100.0,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_bucket_col=grouped_bucket,
            grouped_values_col=grouped_values,
        )

        assert tensor[1, 0, 0] == pytest.approx(10.0 / 100.0)
        assert tensor[0, 1, 0] == pytest.approx(20.0 / 100.0)
        assert tensor[2, 0, 0] == pytest.approx(30.0 / 100.0)
        assert tensor[0, 2, 0] == pytest.approx(40.0 / 100.0)

    def test_unknown_keys_ignored(self):
        anchor_keys = pa.array(["A", "B"])
        grouped_fk = pa.array(["A", "UNKNOWN", "B"])
        grouped_bucket = pa.array([0, 0, 0], type=pa.int64())
        grouped_values = pa.array([1.0, 999.0, 2.0])
        tensor = np.zeros((2, 1, 1), dtype=np.float32)

        vectorized_scatter(
            tensor, dim_idx=0, edge_max=10.0,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_bucket_col=grouped_bucket,
            grouped_values_col=grouped_values,
        )

        assert tensor[0, 0, 0] == pytest.approx(0.1)
        assert tensor[1, 0, 0] == pytest.approx(0.2)

    def test_nan_values_skipped(self):
        anchor_keys = pa.array(["A"])
        grouped_fk = pa.array(["A", "A"])
        grouped_bucket = pa.array([0, 1], type=pa.int64())
        grouped_values = pa.array([5.0, float("nan")])
        tensor = np.zeros((1, 2, 1), dtype=np.float32)

        vectorized_scatter(
            tensor, dim_idx=0, edge_max=10.0,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_bucket_col=grouped_bucket,
            grouped_values_col=grouped_values,
        )

        assert tensor[0, 0, 0] == pytest.approx(0.5)
        assert tensor[0, 1, 0] == 0.0

    def test_clipping(self):
        anchor_keys = pa.array(["A"])
        grouped_fk = pa.array(["A", "A"])
        grouped_bucket = pa.array([0, 1], type=pa.int64())
        grouped_values = pa.array([200.0, -5.0])
        tensor = np.zeros((1, 2, 1), dtype=np.float32)

        vectorized_scatter(
            tensor, dim_idx=0, edge_max=100.0,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_bucket_col=grouped_bucket,
            grouped_values_col=grouped_values,
        )

        assert tensor[0, 0, 0] == pytest.approx(1.0)
        assert tensor[0, 1, 0] == pytest.approx(0.0)

    def test_empty_input(self):
        anchor_keys = pa.array(["A", "B"])
        grouped_fk = pa.array([], type=pa.string())
        grouped_bucket = pa.array([], type=pa.int64())
        grouped_values = pa.array([], type=pa.float64())
        tensor = np.zeros((2, 3, 1), dtype=np.float32)

        vectorized_scatter(
            tensor, dim_idx=0, edge_max=10.0,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_bucket_col=grouped_bucket,
            grouped_values_col=grouped_values,
        )

        assert np.all(tensor == 0.0)

    def test_multiple_dims(self):
        anchor_keys = pa.array(["A", "B"])
        grouped_fk = pa.array(["A", "B"])
        grouped_bucket = pa.array([0, 0], type=pa.int64())
        tensor = np.zeros((2, 1, 3), dtype=np.float32)

        for dim in range(3):
            vals = pa.array([float(dim + 1) * 10, float(dim + 1) * 20])
            vectorized_scatter(
                tensor, dim_idx=dim, edge_max=100.0,
                anchor_keys_arr=anchor_keys,
                grouped_fk_col=grouped_fk,
                grouped_bucket_col=grouped_bucket,
                grouped_values_col=vals,
            )

        assert tensor[0, 0, 0] == pytest.approx(0.1)
        assert tensor[1, 0, 0] == pytest.approx(0.2)
        assert tensor[0, 0, 2] == pytest.approx(0.3)
        assert tensor[1, 0, 2] == pytest.approx(0.6)


class TestVectorizedScatter1D:
    def test_basic_1d(self):
        anchor_keys = pa.array(["X", "Y", "Z"])
        grouped_fk = pa.array(["Z", "X"])
        grouped_values = pa.array([30.0, 10.0])
        result = np.zeros(3, dtype=np.float64)

        vectorized_scatter_1d(
            result,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_values_col=grouped_values,
        )

        assert result[0] == pytest.approx(10.0)
        assert result[1] == 0.0
        assert result[2] == pytest.approx(30.0)

    def test_unknown_keys_1d(self):
        anchor_keys = pa.array(["A", "B"])
        grouped_fk = pa.array(["A", "UNKNOWN"])
        grouped_values = pa.array([5.0, 999.0])
        result = np.zeros(2, dtype=np.float64)

        vectorized_scatter_1d(
            result,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_values_col=grouped_values,
        )

        assert result[0] == pytest.approx(5.0)
        assert result[1] == 0.0

    def test_nan_1d(self):
        anchor_keys = pa.array(["A"])
        grouped_fk = pa.array(["A", "A"])
        grouped_values = pa.array([float("nan"), 7.0])
        result = np.zeros(1, dtype=np.float64)

        vectorized_scatter_1d(
            result,
            anchor_keys_arr=anchor_keys,
            grouped_fk_col=grouped_fk,
            grouped_values_col=grouped_values,
        )

        # Last non-NaN wins (or first — depends on order)
        assert result[0] == pytest.approx(7.0)
