# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Vectorized Arrow→tensor scatter utilities.

Replaces Python-level row iteration with Arrow pc.index_in + numpy
fancy indexing. Used by both static (compute_derived_batch) and
temporal (_precompute_shape_tensor) build paths.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray


def vectorized_scatter(
    tensor: NDArray[np.float32],
    dim_idx: int,
    edge_max: float,
    anchor_keys_arr: pa.Array,
    grouped_fk_col: pa.Array,
    grouped_bucket_col: pa.Array,
    grouped_values_col: pa.Array,
) -> None:
    if len(grouped_fk_col) == 0:
        return

    entity_idx_arr = pc.index_in(grouped_fk_col, anchor_keys_arr)
    valid = pc.and_(pc.is_valid(entity_idx_arr), pc.is_valid(grouped_values_col))

    entity_indices = entity_idx_arr.filter(valid).to_numpy(
        zero_copy_only=False,
    ).astype(np.intp)
    bucket_indices = grouped_bucket_col.filter(valid).to_numpy(
        zero_copy_only=False,
    ).astype(np.intp)
    values = grouped_values_col.filter(valid).to_numpy(
        zero_copy_only=False,
    ).astype(np.float64)

    nan_ok = ~np.isnan(values)
    entity_indices = entity_indices[nan_ok]
    bucket_indices = bucket_indices[nan_ok]
    values = np.clip(values[nan_ok], 0.0, edge_max) / edge_max

    tensor[entity_indices, bucket_indices, dim_idx] = values.astype(np.float32)


def vectorized_scatter_1d(
    result: NDArray[np.float64],
    anchor_keys_arr: pa.Array,
    grouped_fk_col: pa.Array,
    grouped_values_col: pa.Array,
) -> None:
    if len(grouped_fk_col) == 0:
        return

    entity_idx_arr = pc.index_in(grouped_fk_col, anchor_keys_arr)
    valid = pc.and_(pc.is_valid(entity_idx_arr), pc.is_valid(grouped_values_col))

    entity_indices = entity_idx_arr.filter(valid).to_numpy(
        zero_copy_only=False,
    ).astype(np.intp)
    values = grouped_values_col.filter(valid).to_numpy(
        zero_copy_only=False,
    ).astype(np.float64)

    nan_ok = ~np.isnan(values)
    result[entity_indices[nan_ok]] = values[nan_ok]
