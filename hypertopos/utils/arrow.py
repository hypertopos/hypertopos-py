# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def flatten_edges_for_sql(table: pa.Table) -> pa.Table:
    """Add ``edge_line_ids`` and ``edge_point_keys`` flat columns derived from ``edges``.

    Used to prepare a geometry table for DataFusion SQL queries that need
    flat list<string> columns for ``list_position()`` lookups.

    Only alive edges are included in the flat arrays.
    """
    edges_col = table.column("edges").combine_chunks()
    flat_all = pc.list_flatten(edges_col)
    alive_mask = pc.equal(pc.struct_field(flat_all, "status"), "alive")

    flat_line_ids = pc.struct_field(flat_all, "line_id")
    flat_point_keys = pc.struct_field(flat_all, "point_key")

    # Vectorized: filter alive edges, then rebuild per-row lists via offsets
    offsets = edges_col.offsets.to_numpy()
    alive_np = alive_mask.to_numpy(zero_copy_only=False)

    # Count alive edges per row using np.add.reduceat
    n = len(edges_col)
    if len(alive_np) == 0:
        alive_per_row = np.zeros(n, dtype=np.int64)
    else:
        alive_per_row = np.add.reduceat(alive_np.astype(np.int64), offsets[:-1])
        # reduceat on empty slices (offset[i] == offset[i+1]) gives wrong value;
        # fix: zero out rows where the slice was empty
        empty_mask = offsets[:-1] == offsets[1:]
        alive_per_row[empty_mask] = 0

    new_offsets = np.empty(n + 1, dtype=np.int64)
    new_offsets[0] = 0
    np.cumsum(alive_per_row, out=new_offsets[1:])

    # Filter to alive-only flat arrays
    alive_lids = flat_line_ids.filter(alive_mask)
    alive_pkeys = flat_point_keys.filter(alive_mask)

    # Build list arrays from flat + offsets
    offsets_arr = pa.array(new_offsets, type=pa.int64())
    lid_list = pa.ListArray.from_arrays(offsets_arr, alive_lids)
    pkey_list = pa.ListArray.from_arrays(offsets_arr, alive_pkeys)

    table = table.append_column(
        "edge_line_ids", lid_list.cast(pa.list_(pa.string()))
    )
    table = table.append_column(
        "edge_point_keys", pkey_list.cast(pa.list_(pa.string()))
    )
    return table


def reconstruct_edges_from_entity_keys(
    entity_keys_col: pa.ChunkedArray,
    relations: list,
) -> pa.Array:
    """Reconstruct edges struct list from entity_keys + pattern relations.

    entity_keys[i] maps to relations[i] positionally.
    Empty string = dead edge.
    """
    from hypertopos.storage._schemas import EDGE_STRUCT_TYPE

    rows = entity_keys_col.to_pylist()
    edge_lists = []
    for keys in rows:
        edges = []
        if keys:
            for i, rel in enumerate(relations):
                key = keys[i] if i < len(keys) else ""
                edges.append({
                    "line_id": rel.line_id,
                    "point_key": key,
                    "status": "alive" if key else "dead",
                    "direction": rel.direction,
                })
        edge_lists.append(edges)
    return pa.array(edge_lists, type=pa.list_(EDGE_STRUCT_TYPE))


def delta_matrix_from_arrow(table: pa.Table) -> np.ndarray:
    col = table["delta"].combine_chunks()
    n = len(col)
    if n == 0:
        d = col.type.list_size if isinstance(col.type, pa.FixedSizeListType) else 0
        return np.empty((0, d), dtype=np.float32)
    # Works for both FixedSizeListType and ListType:
    # ListArray.values returns the flat contiguous buffer — zero-copy.
    flat = col.values.to_numpy(zero_copy_only=True)
    if col.null_count:
        raise ValueError(
            f"delta column must not contain null rows (got {col.null_count}); "
            "geometry written by GDSWriter is always non-null"
        )
    if len(flat) % n:
        raise ValueError(
            f"delta column has ragged rows: {len(flat)} values / {n} rows is not integer"
        )
    d = len(flat) // n
    return flat.reshape(n, d).astype(np.float32, copy=False)
