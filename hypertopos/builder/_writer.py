# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import warnings
from pathlib import Path

import lance as _lance
import numpy as np
import pyarrow as pa

from hypertopos.storage.writer import GDSWriter, _write_lance


def write_points(
    base_path: Path,
    line_id: str,
    table: pa.Table,
    version: int,
    partition_col: str | None,
    fts_columns: list[str] | str = "all",
) -> None:
    """Write points as Lance dataset and build indices.

    Args:
        fts_columns: Controls which string columns get INVERTED (FTS) indices.
            ``"all"`` — every string column except primary_key (legacy default).
            ``["col1", "col2"]`` — only the listed columns.
            ``[]`` — no INVERTED indices at all.
    """
    base = base_path / "points" / line_id / f"v={version}"
    lance_path = base / "data.lance"
    lance_path.parent.mkdir(parents=True, exist_ok=True)
    ds = _write_lance(table, str(lance_path))
    # Build INVERTED (FTS) indices based on fts_columns setting
    if fts_columns == "all":
        for field in table.schema:
            if pa.types.is_string(field.type) and field.name != "primary_key":
                try:
                    ds.create_scalar_index(field.name, index_type="INVERTED")
                except Exception as exc:
                    warnings.warn(
                        f"INVERTED index skipped for column '{field.name}': {exc}",
                        stacklevel=2,
                    )
    elif isinstance(fts_columns, list) and fts_columns:
        for col_name in fts_columns:
            if col_name in table.schema.names:
                try:
                    ds.create_scalar_index(col_name, index_type="INVERTED")
                except Exception as exc:
                    warnings.warn(
                        f"INVERTED index skipped for column '{col_name}': {exc}",
                        stacklevel=2,
                    )
    # BTREE index on primary_key for fast read_points_batch lookups
    try:
        ds.create_scalar_index("primary_key", index_type="BTREE")
    except Exception as exc:
        warnings.warn(
            f"BTREE index skipped for primary_key in '{line_id}': {exc}",
            stacklevel=2,
        )


def _prepare_geometry_for_lance(table: pa.Table) -> tuple[pa.Table, int]:
    """Cast delta to fixed-size list and add per-dimension scalar columns.

    Returns (prepared_table, list_size).
    """
    list_size = 0
    if table.num_rows > 0:
        delta_col = table["delta"]
        list_size = len(delta_col[0])  # ListScalar supports len() natively
        fixed_type = pa.list_(pa.float32(), list_size)
        fixed_delta = delta_col.cast(fixed_type)
        table = table.set_column(
            table.schema.get_field_index("delta"), "delta", fixed_delta
        )

        # Per-dimension scalar columns for Lance predicate pushdown
        if list_size > 0:
            flat = fixed_delta.combine_chunks().values.to_numpy(
                zero_copy_only=False,
            )
            matrix = flat.reshape(-1, list_size)
            delta_cols = {
                f"delta_dim_{dim_idx}": pa.array(matrix[:, dim_idx], type=pa.float32())
                for dim_idx in range(list_size)
            }
            existing = {name: table.column(name) for name in table.schema.names}
            existing.update(delta_cols)
            table = pa.table(existing)
    return table, list_size


def write_geometry(
    base_path: Path,
    pattern_id: str,
    table: pa.Table,
    version: int,
) -> None:
    """Write geometry Lance dataset for a pattern. Sorts by delta_norm descending.

    Casts delta to a fixed-size list (required for IVF_FLAT ANN indexing), then
    writes the dataset and builds IVF_FLAT + scalar indices via GDSWriter.

    Auto-detects binary geometry (all shape values in {0, 1}) and skips IVF_FLAT
    index — binary vectors produce degenerate KMeans clusters, wasting 30s+.
    """
    sorted_table = table.sort_by([("delta_norm", "descending")])
    sorted_table, list_size = _prepare_geometry_for_lance(sorted_table)

    # Detect binary/low-cardinality geometry — skip IVF_FLAT if unique vector
    # count is very low relative to population. Binary shape vectors (0/1)
    # produce at most 2^D unique deltas; KMeans with 256 partitions wastes 30s+.
    skip_vector = False
    if sorted_table.num_rows > 0 and list_size > 0:
        sample_size = min(sorted_table.num_rows, 5000)
        sample_tbl = sorted_table.slice(0, sample_size)
        delta_flat = sample_tbl["delta"].combine_chunks().values.to_numpy(
            zero_copy_only=False,
        )
        sample_deltas = delta_flat.reshape(len(sample_tbl), -1)
        n_unique = len(np.unique(sample_deltas, axis=0))
        if n_unique < 256:
            skip_vector = True

    lance_path = base_path / "geometry" / pattern_id / f"v={version}" / "data.lance"
    lance_path.parent.mkdir(parents=True, exist_ok=True)
    ds = _write_lance(sorted_table, str(lance_path))

    writer = GDSWriter(str(base_path))
    writer.build_index_if_needed(
        pattern_id, version, _ds=ds, skip_vector_index=skip_vector,
    )


def write_geometry_chunk(
    base_path: Path,
    pattern_id: str,
    table: pa.Table,
    version: int,
) -> None:
    """Append a geometry chunk to the Lance dataset (no sorting, no indexing).

    Used by chunked geometry builds. Caller must call
    finalize_geometry_chunks() after all chunks are written.
    """
    prepared, _list_size = _prepare_geometry_for_lance(table)

    lance_path = (
        base_path / "geometry" / pattern_id / f"v={version}" / "data.lance"
    )
    lance_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "append" if lance_path.exists() else "create"
    _write_lance(prepared, str(lance_path), mode=mode)


def finalize_geometry_chunks(
    base_path: Path,
    pattern_id: str,
    version: int,
) -> None:
    """Compact fragments and build indices after chunked geometry writes.

    Call once after all write_geometry_chunk() calls for a pattern.
    """
    lance_path = (
        base_path / "geometry" / pattern_id / f"v={version}" / "data.lance"
    )
    if not lance_path.exists():
        return
    ds = _lance.dataset(str(lance_path))
    n_rows = ds.count_rows()
    if n_rows == 0:
        return

    # Compact fragments from multiple appends into fewer files.
    # target_rows_per_fragment=1M merges small chunks into large fragments
    # for better sequential scan performance (fewer random I/O ops).
    ds.optimize.compact_files(target_rows_per_fragment=1_048_576)
    ds = _lance.dataset(str(lance_path))  # re-open after compaction

    # Detect list_size for index building
    delta_field = ds.schema.field("delta")
    list_size: int | None = None
    if hasattr(delta_field.type, "list_size"):
        list_size = delta_field.type.list_size

    # Detect binary/low-cardinality — skip IVF_FLAT
    skip_vector = False
    if list_size is not None and n_rows > 0:
        sample_size = min(n_rows, 5000)
        sample_tbl = ds.head(sample_size, columns=["delta"])
        delta_col = sample_tbl["delta"].combine_chunks()
        # ds.head may return variable-length list; use field metadata for reshape
        effective_list_size = list_size
        if hasattr(delta_col.type, "list_size") and delta_col.type.list_size:
            effective_list_size = delta_col.type.list_size
        delta_flat = delta_col.values.to_numpy(zero_copy_only=False)
        sample_deltas = delta_flat.reshape(len(sample_tbl), effective_list_size)
        n_unique = len(np.unique(sample_deltas, axis=0))
        if n_unique < 256:
            skip_vector = True

    writer = GDSWriter(str(base_path))
    writer.build_index_if_needed(
        pattern_id, version, _ds=ds, skip_vector_index=skip_vector,
    )


def update_points(
    base_path: Path,
    line_id: str,
    table: pa.Table,
    version: int,
) -> None:
    """Upsert entity rows in points Lance dataset."""
    lance_path = base_path / "points" / line_id / f"v={version}" / "data.lance"
    if not lance_path.exists():
        raise FileNotFoundError(
            f"Points dataset not found at {lance_path}. "
            "Run build() first to create the initial dataset."
        )
    ds = _lance.dataset(str(lance_path))
    ds.merge_insert("primary_key") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute(table)


def delete_points(
    base_path: Path,
    line_id: str,
    keys: list[str],
    version: int,
) -> None:
    """Delete entity rows from points Lance dataset."""
    lance_path = base_path / "points" / line_id / f"v={version}" / "data.lance"
    if not lance_path.exists():
        return
    ds = _lance.dataset(str(lance_path))
    escaped = [k.replace("'", "''") for k in keys]
    in_clause = ", ".join(f"'{k}'" for k in escaped)
    ds.delete(f"primary_key IN ({in_clause})")
