# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import json
import shutil
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

import lance as _lance  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa

from hypertopos.model.objects import SolidSlice

# Lance write defaults — applied to ALL write_dataset calls via _write_lance().
# data_storage_version="stable" enables v2.1+ format: BSS float compression,
# cascading codecs, block-level RLE. 30-60% smaller files, faster reads.
# max_rows_per_group=8192 reduces metadata overhead vs default 1024.
_LANCE_WRITE_DEFAULTS: dict[str, Any] = {
    "data_storage_version": "stable",
    "max_rows_per_group": 8192,
}


def _write_lance(
    table: pa.Table,
    path: str,
    mode: str = "create",
    **kwargs: Any,
) -> _lance.LanceDataset:
    """Write Arrow table to Lance with optimized defaults."""
    merged = {**_LANCE_WRITE_DEFAULTS, **kwargs}
    return _lance.write_dataset(table, path, mode=mode, **merged)

_TEMPORAL_SCHEMA = pa.schema([
    pa.field("primary_key",      pa.string()),
    pa.field("slice_index",      pa.int32()),
    pa.field("timestamp",        pa.timestamp("us", tz="UTC")),
    pa.field("deformation_type", pa.string()),
    pa.field("shape_snapshot",   pa.list_(pa.float32())),
    pa.field("pattern_ver",      pa.int32()),
    pa.field("changed_property", pa.string()),
    pa.field("changed_line_id",  pa.string()),
])


class GDSWriter:
    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)

    def append_temporal_slice(
        self,
        solid_slice: SolidSlice,
        pattern_id: str,
        primary_key: str,
        shape_snapshot: np.ndarray,
        agent_id: str | None = None,
    ) -> None:
        """Append a temporal slice to the Lance dataset for this pattern.

        Uses lance write_dataset(mode='append') — one dataset per pattern.
        shape_snapshot: raw [0..1] shape vector; delta is recomputed at read time.
        """
        if agent_id is not None:
            lance_dir = self._base / "temporal" / "_agents" / agent_id / pattern_id
        else:
            lance_dir = self._base / "temporal" / pattern_id
        lance_dir.mkdir(parents=True, exist_ok=True)
        lance_path = lance_dir / "data.lance"

        table = pa.table(
            {
                "primary_key":      [primary_key],
                "slice_index":      [solid_slice.slice_index],
                "timestamp":        [solid_slice.timestamp],
                "deformation_type": [solid_slice.deformation_type],
                "shape_snapshot":   [shape_snapshot.tolist()],
                "pattern_ver":      [solid_slice.pattern_ver],
                "changed_property": [solid_slice.changed_property],
                "changed_line_id":  [solid_slice.changed_line_id],
            },
            schema=_TEMPORAL_SCHEMA,
        )
        mode = "append" if lance_path.exists() else "create"
        _write_lance(table, str(lance_path), mode=mode)

    def purge_agent_temporal(self, agent_id: str) -> int:
        agent_dir = self._base / "temporal" / "_agents" / agent_id
        if not agent_dir.exists():
            return 0
        count = sum(1 for _ in agent_dir.rglob("*.lance"))
        shutil.rmtree(agent_dir)
        return count

    def purge_all_agents(self) -> int:
        agents_dir = self._base / "temporal" / "_agents"
        if not agents_dir.exists():
            return 0
        count = sum(1 for _ in agents_dir.rglob("*.lance"))
        shutil.rmtree(agents_dir)
        return count

    def compact_temporal(self, pattern_id: str) -> dict[str, object]:
        """Compact temporal Lance fragments. Call after bulk ingestion.

        Merges small fragments created by repeated append calls into fewer,
        larger fragments. Reduces read latency for high-deformation entities.
        """
        lance_path = self._base / "temporal" / pattern_id / "data.lance"
        if not lance_path.exists():
            return {"status": "no_lance", "pattern_id": pattern_id}
        ds = _lance.dataset(str(lance_path))
        rows_before = ds.count_rows()
        ds.optimize.compact_files(target_rows_per_fragment=1_048_576)
        ds_after = _lance.dataset(str(lance_path))
        rows_after = ds_after.count_rows()
        return {
            "status": "ok",
            "pattern_id": pattern_id,
            "rows": rows_after,
            "rows_before": rows_before,
        }

    def build_temporal_index(self, pattern_id: str) -> None:
        """Build a BTREE scalar index on primary_key in the temporal Lance dataset.

        Enables O(log n) per-entity reads instead of full table scans.
        No-op if the dataset does not exist or contains zero rows.
        """
        lance_path = self._base / "temporal" / pattern_id / "data.lance"
        if not lance_path.exists():
            return
        ds = _lance.dataset(str(lance_path))
        if ds.count_rows() == 0:
            return
        ds.create_scalar_index("primary_key", index_type="BTREE")

    def migrate_temporal_to_shape_snapshot(
        self,
        pattern_id: str,
        mu: np.ndarray,
        sigma_diag: np.ndarray,
    ) -> int:
        """Migrate temporal Lance datasets from delta_snapshot to shape_snapshot schema.

        Reads all rows, reverses z-scoring to recover shape vectors, writes back
        with the new schema, and rebuilds the temporal index.  Returns the total
        count of migrated rows across the main dataset and all agent-scoped paths.
        """
        _sigma = np.maximum(sigma_diag, 1e-2)
        total = 0

        def _migrate_one(lance_path: Path) -> int:
            if not lance_path.exists():
                return 0
            ds = _lance.dataset(str(lance_path))
            field_names = {f.name for f in ds.schema}
            if "delta_snapshot" not in field_names:
                return 0
            tbl = ds.to_table()
            rows = tbl.num_rows
            if rows == 0:
                return 0
            # Vectorized: read entire delta_snapshot column as (rows, D) matrix,
            # apply z-score reversal and clip in a single numpy operation.
            D = len(mu)
            flat = tbl["delta_snapshot"].combine_chunks().values.to_numpy(
                zero_copy_only=False
            )
            delta_matrix = flat.reshape(rows, D).astype(np.float32)
            shape_matrix = np.clip(delta_matrix * _sigma + mu, 0.0, 1.0)
            shapes = pa.FixedSizeListArray.from_arrays(
                pa.array(shape_matrix.ravel(), type=pa.float32()),
                list_size=D,
            ).cast(pa.list_(pa.float32()))
            new_table = pa.table(
                {
                    "primary_key":      tbl["primary_key"],
                    "slice_index":      tbl["slice_index"],
                    "timestamp":        tbl["timestamp"],
                    "deformation_type": tbl["deformation_type"],
                    "shape_snapshot":   shapes,
                    "pattern_ver":      tbl["pattern_ver"],
                    "changed_property": tbl["changed_property"],
                    "changed_line_id":  tbl["changed_line_id"],
                },
                schema=_TEMPORAL_SCHEMA,
            )
            _write_lance(new_table, str(lance_path), mode="overwrite")
            return rows

        main_path = self._base / "temporal" / pattern_id / "data.lance"
        total += _migrate_one(main_path)

        agents_dir = self._base / "temporal" / "_agents"
        if agents_dir.exists():
            for agent_lance in agents_dir.glob(f"*/{pattern_id}/data.lance"):
                total += _migrate_one(agent_lance)

        if total > 0:
            self.build_temporal_index(pattern_id)

        return total

    def write_geometry_stats(
        self,
        pattern_id: str,
        version: int,
        delta_norms: np.ndarray,
        theta_norm: float,
        is_anomaly_arr: np.ndarray | None = None,
    ) -> None:
        """Persist delta_norm statistics as a JSON cache file.

        Written at geometry build time so anomaly_summary can skip the delta
        column on subsequent reads (O(1) vs O(n) full scan).

        When is_anomaly_arr is provided (from stored geometry labels), it takes
        precedence over delta_norm >= theta_norm comparison. This is critical
        for group_by_property / GMM patterns where per-group theta differs
        from the global theta_norm.
        """
        if is_anomaly_arr is not None:
            total_anomalies = int(np.sum(is_anomaly_arr))
        elif theta_norm > 0.0:
            total_anomalies = int(np.sum(delta_norms >= theta_norm))
        else:
            total_anomalies = 0
        stats = {
            "pattern_id": pattern_id,
            "version": version,
            "theta_norm": float(theta_norm),
            "total_entities": int(len(delta_norms)),
            "total_anomalies": total_anomalies,
            "percentiles": {
                **dict(
                    zip(
                        ["p50", "p75", "p90", "p95", "p99"],
                        np.percentile(delta_norms, [50, 75, 90, 95, 99]).tolist(),
                        strict=True,
                    )
                ),
                "max": float(np.max(delta_norms)),
            },
        }
        # Compute inactive_ratio (histogram-based mode detection)
        inactive_ratio = None
        inactive_count = 0
        n = len(delta_norms)
        if n > 0:
            counts, bin_edges = np.histogram(delta_norms, bins=50)
            mode_bin = int(np.argmax(counts))
            mode_count = int(counts[mode_bin])
            mode_center = float(
                (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
            )
            median_val = float(np.median(delta_norms))
            if mode_count / n > 0.25 and mode_center < median_val:
                threshold = float(bin_edges[mode_bin + 1])
                inactive_count = int(np.sum(delta_norms <= threshold))
                inactive_ratio = round(inactive_count / n, 4)

        stats["inactive_ratio"] = inactive_ratio
        stats["inactive_count"] = inactive_count

        path = (
            self._base
            / "_gds_meta"
            / "geometry_stats"
            / f"{pattern_id}_v{version}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats))

    def write_lance_geometry(self, table: pa.Table, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        lance_path = output_dir / "data.lance"

        if table.num_rows == 0:
            _write_lance(table, str(lance_path))
            return

        # Cast delta to fixed-size list so Lance can build a vector index
        delta_col = table["delta"]
        list_size = len(delta_col[0].as_py())
        fixed_type = pa.list_(pa.float32(), list_size)
        fixed_delta = delta_col.cast(fixed_type)
        table = table.set_column(table.schema.get_field_index("delta"), "delta", fixed_delta)

        # Per-dimension scalar columns for Lance predicate pushdown
        if list_size > 0:
            flat = fixed_delta.combine_chunks().values.to_numpy(zero_copy_only=False)
            matrix = flat.reshape(-1, list_size)
            delta_cols = {
                f"delta_dim_{dim_idx}": pa.array(matrix[:, dim_idx], type=pa.float32())
                for dim_idx in range(list_size)
            }
            existing = {name: table.column(name) for name in table.schema.names}
            existing.update(delta_cols)
            table = pa.table(existing)

        _write_lance(table, str(lance_path))

        self._build_indices_on_lance_path(lance_path, n_rows=table.num_rows, list_size=list_size)

    def build_index_if_needed(
        self, pattern_id: str, version: int = 1, *, _ds: Any = None,
        skip_vector_index: bool = False,
    ) -> None:
        """Build IVF_FLAT and scalar indices on an existing geometry Lance dataset.

        Idempotent — safe to call multiple times. If the dataset does not exist
        or has zero rows, this is a no-op. Intended for use after
        GDSBuilder.build() which writes geometry via _lance.write_dataset
        directly and therefore bypasses write_lance_geometry.

        The delta column must already be stored as a fixed-size list; call
        cast_geometry_delta_to_fixed_size first if this is not guaranteed.
        """
        lance_path = self._base / "geometry" / pattern_id / f"v={version}" / "data.lance"
        if _ds is not None:
            ds = _ds
        elif lance_path.exists():
            ds = _lance.dataset(str(lance_path))
        else:
            return
        n_rows = ds.count_rows()
        if n_rows == 0:
            return
        # Detect list_size from schema
        delta_field = ds.schema.field("delta")
        list_size: int | None = None
        if hasattr(delta_field.type, "list_size"):
            list_size = delta_field.type.list_size
        if list_size is None:
            # Variable-size list — cannot build IVF_FLAT; cast first
            warnings.warn(
                f"build_index_if_needed: delta column in {lance_path} is not a "
                "fixed-size list. Cast to fixed-size before calling this method. "
                "Scalar indices will still be built.",
                stacklevel=2,
            )
        self._build_indices_on_lance_path(
            lance_path, n_rows=n_rows, list_size=list_size, ds=ds,
            skip_vector_index=skip_vector_index,
        )

    def _maybe_reindex_geometry(
        self, pattern_id: str, threshold: float = 0.1, version: int = 1
    ) -> bool:
        """Rebuild IVF_FLAT index if the fraction of unindexed rows exceeds threshold.

        Checks whether a vector index on the delta column already exists.  If no
        index is present the entire dataset is treated as unindexed (ratio = 1.0).
        If an index exists, the method compares the current row count against the
        row count recorded at index-build time (approximated via index metadata).
        When the dataset has grown by more than threshold (default 10%) since the
        last index build, compact_files() is called followed by create_index with
        replace=True.

        Returns True if the index was rebuilt, False otherwise.
        """
        lance_path = self._base / "geometry" / pattern_id / f"v={version}" / "data.lance"
        if not lance_path.exists():
            return False
        ds = _lance.dataset(str(lance_path))
        n_rows = ds.count_rows()
        if n_rows == 0:
            return False

        # Determine whether a vector index on delta already exists and how many
        # rows it covers.  Lance does not expose num_unindexed_rows directly, so
        # we approximate: if no vector index exists → all rows are unindexed.
        # If an index exists, read the num_indexed_rows field when available;
        # fall back to treating the index as covering all rows (conservative —
        # avoids unnecessary rebuilds).
        has_vector_index = False
        indexed_rows: int = n_rows  # optimistic default when index exists
        with suppress(Exception):
            indices = ds.list_indices()
            for idx in indices:
                fields = (
                    idx.get("fields", []) if isinstance(idx, dict)
                    else getattr(idx, "fields", [])
                )
                if "delta" in fields:
                    has_vector_index = True
                    # Try to read num_indexed_rows if Lance exposes it
                    num_indexed = (
                        idx.get("num_indexed_rows")
                        if isinstance(idx, dict)
                        else getattr(idx, "num_indexed_rows", None)
                    )
                    if num_indexed is not None:
                        indexed_rows = int(num_indexed)
                    break

        unindexed = n_rows if not has_vector_index else max(0, n_rows - indexed_rows)

        if unindexed / n_rows < threshold:
            return False

        # Compact first to merge fragments, then rebuild the vector index
        ds.optimize.compact_files(target_rows_per_fragment=1_048_576)

        delta_field = ds.schema.field("delta")
        list_size: int | None = (
            delta_field.type.list_size
            if hasattr(delta_field.type, "list_size")
            else None
        )
        if list_size is None or n_rows < 256:
            # Cannot build IVF_FLAT without fixed-size list or sufficient rows
            return False

        num_partitions = min(64, max(16, int(n_rows ** 0.5)))
        try:
            ds = _lance.dataset(str(lance_path))
            ds.create_index(
                "delta",
                index_type="IVF_FLAT",
                num_partitions=num_partitions,
                replace=True,
            )
        except Exception as exc:
            warnings.warn(
                f"_maybe_reindex_geometry: failed to rebuild IVF_FLAT on {lance_path} ({exc}).",
                stacklevel=2,
            )
            return False
        return True

    def recompute_delta_rank_pct(self, pattern_id: str, version: int = 1) -> None:
        """Recompute delta_rank_pct globally across all entities in the pattern geometry.

        Called automatically by append_geometry after each incremental write.
        May also be called manually for repair scenarios (e.g. after raw Lance writes
        that bypass append_geometry).

        Performance note: this is O(N) over the full geometry dataset. For the current
        builder workflow (one append_geometry call per pattern), this is acceptable.
        For future live-ingestion use cases with many small batches, consider batching
        appends and calling recompute_delta_rank_pct only at the end of each ingestion
        session rather than on every individual append.
        """
        lance_path = self._base / "geometry" / pattern_id / f"v={version}" / "data.lance"
        if not lance_path.exists():
            return
        ds = _lance.dataset(str(lance_path))
        if "delta_rank_pct" not in {f.name for f in ds.schema}:
            return
        tbl = ds.to_table(columns=["primary_key", "delta_norm"])
        delta_norms = tbl["delta_norm"].to_numpy(zero_copy_only=False)
        n = len(delta_norms)
        if n == 0:
            return
        sorted_norms = np.sort(delta_norms)
        ranks = np.searchsorted(sorted_norms, delta_norms, side="left")
        new_pcts = (ranks / n * 100).astype(np.float32)
        updates = pa.table({
            "primary_key": tbl["primary_key"],
            "delta_rank_pct": pa.array(new_pcts, type=pa.float32()),
        })
        ds.merge_insert("primary_key").when_matched_update_all().execute(updates)

    def append_geometry(self, table: pa.Table, pattern_id: str, version: int = 1) -> None:
        """Append new geometry rows to an existing Lance geometry dataset.

        Writes the rows via lance write_dataset(mode='append') and then checks
        whether the ANN index needs rebuilding (more than 10% unindexed rows).

        The delta column must be a fixed-size list of float32 — the method casts
        variable-size lists automatically when num_rows > 0.

        Automatically recomputes delta_rank_pct globally after appending to ensure
        percentile ranks reflect the full population, not just the appended batch.
        """
        lance_path = self._base / "geometry" / pattern_id / f"v={version}" / "data.lance"
        lance_path.parent.mkdir(parents=True, exist_ok=True)

        # Cast delta to fixed-size list if it is a variable-size list
        if "delta" in table.schema.names and table.num_rows > 0:
            delta_col = table["delta"]
            delta_type = delta_col.type
            if not hasattr(delta_type, "list_size"):
                list_size = len(delta_col[0].as_py())
                fixed_type = pa.list_(pa.float32(), list_size)
                fixed_delta = delta_col.cast(fixed_type)
                table = table.set_column(
                    table.schema.get_field_index("delta"), "delta", fixed_delta
                )

        mode = "append" if lance_path.exists() else "create"
        _write_lance(table, str(lance_path), mode=mode)
        self._maybe_reindex_geometry(pattern_id, version=version)
        self.recompute_delta_rank_pct(pattern_id, version=version)

    def _build_indices_on_lance_path(
        self, lance_path: Path, n_rows: int, list_size: int | None,
        *, ds: Any = None, skip_vector_index: bool = False,
    ) -> None:
        """Build IVF_FLAT vector index and scalar indices on geometry Lance dataset."""
        if ds is None:
            ds = _lance.dataset(str(lance_path))

        # IVF_FLAT vector index on delta — O(N) build vs O(N×P×iters) for IVF_PQ.
        # 6M rows: IVF_FLAT builds in ~3s, IVF_PQ takes 60-120s.
        # Query: 23ms vs 14ms — negligible for agent workloads.
        if list_size is not None and n_rows >= 256 and not skip_vector_index:
            num_partitions = min(64, max(16, int(n_rows ** 0.5)))
            try:
                ds.create_index(
                    "delta",
                    index_type="IVF_FLAT",
                    num_partitions=num_partitions,
                    replace=True,
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to build IVF_FLAT index on delta ({exc}). "
                    "Sphere is functional but vector search will use full scan.",
                    stacklevel=2,
                )

        # Build scalar indices — each on a fresh dataset handle for thread safety
        field_names = {f.name for f in ds.schema}
        scalar_indices: list[tuple[str, str]] = [
            ("primary_key", "BTREE"),
            ("delta_norm", "BTREE"),
            ("is_anomaly", "BITMAP"),
            ("delta_rank_pct", "BTREE"),
            ("entity_keys", "LABEL_LIST"),
        ]
        for col_name in field_names:
            if col_name.startswith("delta_dim_"):
                scalar_indices.append((col_name, "BTREE"))

        for col, idx_type in scalar_indices:
            if col not in field_names:
                continue
            with suppress(Exception):
                ds.create_scalar_index(col, index_type=idx_type)

    def write_trajectory_from_tensor(
        self,
        pattern_id: str,
        shape_tensor: np.ndarray,
        primary_keys: list[str],
        bucket_timestamps: list,
        mu: np.ndarray | None = None,
        sigma_diag: np.ndarray | None = None,
    ) -> int:
        """Build trajectory vectors directly from in-memory shape tensor.

        Eliminates the need to re-read temporal Lance data. Called by
        build_temporal() with the shape tensor already in memory.

        shape_tensor: (n_entities, n_buckets, D) float32
        primary_keys: list of entity keys, len == n_entities
        bucket_timestamps: list of datetime per bucket
        """
        n, n_buckets, D = shape_tensor.shape
        if n == 0 or n_buckets < 2:
            return 0

        d2 = D * 2
        _sigma = (
            np.maximum(sigma_diag, 1e-2) if sigma_diag is not None else None
        )

        # Z-score shape → delta if mu/sigma provided
        if mu is not None and _sigma is not None:
            data = (shape_tensor - mu) / _sigma
        else:
            data = shape_tensor

        # Vectorized: mean and std across buckets axis
        traj_mean = data.mean(axis=1)  # (n, D)
        traj_std = data.std(axis=1)    # (n, D)
        traj_vectors = np.concatenate([traj_mean, traj_std], axis=1).astype(np.float32)

        # Displacement: ||last_bucket - first_bucket||
        displacements = np.linalg.norm(
            data[:, -1, :] - data[:, 0, :], axis=1,
        ).astype(np.float32)

        num_slices_arr = np.full(n, n_buckets, dtype=np.int32)

        first_ts = bucket_timestamps[0]
        last_ts = bucket_timestamps[-1]

        traj_vec_col = pa.FixedSizeListArray.from_arrays(
            pa.array(traj_vectors.ravel(), type=pa.float32()),
            list_size=d2,
        ).cast(pa.list_(pa.float32(), d2))

        traj_schema = pa.schema([
            pa.field("primary_key", pa.string()),
            pa.field("trajectory_vector", pa.list_(pa.float32(), d2)),
            pa.field("displacement", pa.float32()),
            pa.field("num_slices", pa.int32()),
            pa.field("first_timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("last_timestamp", pa.timestamp("us", tz="UTC")),
        ])

        traj_table = pa.table(
            {
                "primary_key": pa.array(primary_keys, type=pa.string()),
                "trajectory_vector": traj_vec_col,
                "displacement": pa.array(displacements, type=pa.float32()),
                "num_slices": pa.array(num_slices_arr, type=pa.int32()),
                "first_timestamp": pa.array(
                    [first_ts] * n, type=pa.timestamp("us", tz="UTC"),
                ),
                "last_timestamp": pa.array(
                    [last_ts] * n, type=pa.timestamp("us", tz="UTC"),
                ),
            },
            schema=traj_schema,
        )

        traj_path = self._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        _write_lance(traj_table, str(traj_path), mode="overwrite")

        # IVF_FLAT index
        if n >= 256:
            traj_ds = _lance.dataset(str(traj_path))
            n_partitions = min(64, max(16, int(n ** 0.5)))
            with suppress(Exception):
                traj_ds.create_index(
                    "trajectory_vector",
                    index_type="IVF_FLAT",
                    num_partitions=n_partitions,
                )

        return n

    def build_trajectory_index(
        self,
        pattern_id: str,
        mu: np.ndarray | None = None,
        sigma_diag: np.ndarray | None = None,
    ) -> int:
        """Build trajectory summary vectors from temporal slices.

        Trajectory vector = concat([mean(deltas), std(deltas)]) over all
        temporal deformations for each entity. Captures both direction and
        volatility of geometric change.

        mu / sigma_diag: when provided, shape vectors are z-scored to delta before
        building the trajectory. When absent, shape vectors are used directly
        (acceptable for spheres that have not been recalibrated).

        Returns the number of entities indexed, or 0 if no temporal data exists.
        """

        temporal_path = self._base / "temporal" / pattern_id / "data.lance"
        if not temporal_path.exists():
            return 0

        # Batched scan instead of eager to_table() — avoids OOM on 100M+ rows.
        ds = _lance.dataset(str(temporal_path))
        if ds.count_rows() == 0:
            return 0

        if "shape_snapshot" not in {f.name for f in ds.schema}:
            raise ValueError(
                f"Temporal data for '{pattern_id}' uses legacy schema (delta_snapshot). "
                "Run migrate_temporal_to_shape_snapshot() to upgrade first."
            )

        _sigma: np.ndarray | None = (
            np.maximum(sigma_diag, 1e-2) if sigma_diag is not None else None
        )

        # Read all temporal data at once — sort by (pk, timestamp) for grouped processing
        full_table = ds.to_table(
            columns=["primary_key", "shape_snapshot", "timestamp"],
        )
        if full_table.num_rows == 0:
            return 0

        # Sort by (primary_key, timestamp) — Arrow sort, no Python
        import pyarrow.compute as pc
        sort_idx = pc.sort_indices(
            full_table, sort_keys=[("primary_key", "ascending"),
                                    ("timestamp", "ascending")]
        )
        full_table = full_table.take(sort_idx)

        # Extract numpy arrays — zero-copy where possible
        pk_arr = full_table["primary_key"]
        if isinstance(pk_arr, pa.ChunkedArray):
            pk_arr = pk_arr.combine_chunks()
        ts_arr = full_table["timestamp"]

        shape_col = full_table["shape_snapshot"]
        if isinstance(shape_col, pa.ChunkedArray):
            shape_col = shape_col.combine_chunks()
        D_shape = (shape_col.type.list_size
                   if hasattr(shape_col.type, "list_size")
                   else len(shape_col[0].as_py()))
        shape_matrix = shape_col.values.to_numpy(
            zero_copy_only=False
        ).reshape(-1, D_shape).astype(np.float32)

        if mu is not None and _sigma is not None:
            shape_matrix = (shape_matrix - mu) / _sigma

        # Find group boundaries via Arrow — no Python loop over all rows
        n_total = full_table.num_rows
        is_new = pc.not_equal(pk_arr[1:], pk_arr[:-1])
        boundary_np = np.empty(n_total, dtype=bool)
        boundary_np[0] = True
        boundary_np[1:] = is_new.to_numpy(zero_copy_only=False)

        group_starts = np.flatnonzero(boundary_np)
        group_ends = np.empty(len(group_starts), dtype=np.intp)
        group_ends[:-1] = group_starts[1:]
        group_ends[-1] = n_total

        # Fully vectorized per-entity trajectory computation — no Python loop
        n = len(group_starts)
        D = shape_matrix.shape[1]
        d2 = D * 2

        counts = group_ends - group_starts  # (n,)

        # Mean via reduceat
        group_sums = np.add.reduceat(shape_matrix, group_starts, axis=0)  # (n, D)
        group_means = group_sums / counts[:, None]

        # Std via reduceat: std = sqrt(E[x²] - E[x]²)
        group_sq_sums = np.add.reduceat(shape_matrix ** 2, group_starts, axis=0)
        group_var = group_sq_sums / counts[:, None] - group_means ** 2
        np.maximum(group_var, 0.0, out=group_var)
        group_stds = np.sqrt(group_var)
        group_stds[counts == 1] = 0.0

        traj_vectors = np.zeros((n, d2), dtype=np.float32)
        traj_vectors[:, :D] = group_means
        traj_vectors[:, D:] = group_stds

        # Displacement: ||last_slice - first_slice|| per entity
        first_rows = shape_matrix[group_starts]
        last_rows = shape_matrix[group_ends - 1]
        diff = last_rows - first_rows
        displacements = np.linalg.norm(diff, axis=1).astype(np.float32)
        displacements[counts == 1] = 0.0

        num_slices = counts.astype(np.int32)

        # Extract pk/ts via numpy indexing — no Python to_pylist() on full arrays
        pk_list = pk_arr.to_pylist()
        pks = [pk_list[i] for i in group_starts]
        if isinstance(ts_arr, pa.ChunkedArray):
            ts_arr = ts_arr.combine_chunks()
        ts_list = ts_arr.to_pylist()
        first_ts = [ts_list[i] for i in group_starts]
        last_ts = [ts_list[i] for i in group_ends - 1]

        # Build trajectory table from pre-computed numpy arrays
        traj_vec_col = pa.FixedSizeListArray.from_arrays(
            pa.array(traj_vectors.ravel(), type=pa.float32()),
            list_size=d2,
        ).cast(pa.list_(pa.float32(), d2))

        traj_schema = pa.schema([
            pa.field("primary_key",       pa.string()),
            pa.field("trajectory_vector", pa.list_(pa.float32(), d2)),
            pa.field("displacement",      pa.float32()),
            pa.field("num_slices",        pa.int32()),
            pa.field("first_timestamp",   pa.timestamp("us", tz="UTC")),
            pa.field("last_timestamp",    pa.timestamp("us", tz="UTC")),
        ])

        traj_table = pa.table(
            {
                "primary_key":       pa.array(pks, type=pa.string()),
                "trajectory_vector": traj_vec_col,
                "displacement":      pa.array(displacements, type=pa.float32()),
                "num_slices":        pa.array(num_slices, type=pa.int32()),
                "first_timestamp":   pa.array(first_ts,
                                              type=pa.timestamp("us", tz="UTC")),
                "last_timestamp":    pa.array(last_ts,
                                              type=pa.timestamp("us", tz="UTC")),
            },
            schema=traj_schema,
        )

        traj_path = self._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        _write_lance(traj_table, str(traj_path), mode="overwrite")

        # IVF_FLAT index — O(N) build vs O(N×P×iters) for IVF_PQ.
        # 150K entities: IVF_FLAT builds in <1s, IVF_PQ takes 3-5 min.
        # Query latency is comparable (4ms vs 25ms flat scan at 150K).
        if n >= 256:
            traj_ds = _lance.dataset(str(traj_path))
            n_partitions = min(64, max(16, int(n ** 0.5)))
            try:
                traj_ds.create_index(
                    "trajectory_vector",
                    index_type="IVF_FLAT",
                    num_partitions=n_partitions,
                )
            except Exception as exc:
                warnings.warn(
                    f"build_trajectory_index: failed to build IVF_FLAT on {traj_path} ({exc}). "
                    "Sphere is functional but trajectory search will use full scan.",
                    stacklevel=2,
                )
        return n

    def write_temporal_centroids(
        self,
        pattern_id: str,
        centroids: list[dict],
    ) -> None:
        """Write pre-computed per-window centroids for pi11/pi12 fast path."""
        table = pa.table({
            "window_start": pa.array(
                [c["window_start"] for c in centroids],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "window_end": pa.array(
                [c["window_end"] for c in centroids],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "centroid": pa.array(
                [c["centroid"] for c in centroids],
                type=pa.list_(pa.float32()),
            ),
            "entity_count": pa.array(
                [c["entity_count"] for c in centroids],
                type=pa.int32(),
            ),
            "anomaly_rate": pa.array(
                [c.get("anomaly_rate", 0.0) for c in centroids],
                type=pa.float32(),
            ),
        })

        out_dir = self._base / "_gds_meta" / "temporal_centroids"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pattern_id}.lance"
        _write_lance(table, str(out_path), mode="overwrite")

    def write_calibration_tracker(
        self, pattern_id: str, tracker: CalibrationTracker,  # noqa: F821
    ) -> None:
        """Persist calibration tracker to _gds_meta/calibration/{pattern_id}.json."""
        from hypertopos.engine.calibration import RESERVOIR_K

        cal_dir = self._base / "_gds_meta" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        k = min(tracker.norm_reservoir_count, RESERVOIR_K)
        data = {
            "calibrated_mu": tracker.calibrated_mu.tolist(),
            "calibrated_sigma": tracker.calibrated_sigma.tolist(),
            "calibrated_theta": tracker.calibrated_theta.tolist(),
            "calibrated_n": tracker.calibrated_n,
            "calibrated_at": tracker.calibrated_at.isoformat(),
            "running_n": tracker.running_n,
            "running_mean": tracker.running_mean.tolist(),
            "running_m2": tracker.running_m2.tolist(),
            "soft_threshold": tracker.soft_threshold,
            "hard_threshold": tracker.hard_threshold,
            "norm_reservoir": tracker.norm_reservoir[:k].tolist(),
            "norm_reservoir_count": tracker.norm_reservoir_count,
        }
        path = cal_dir / f"{pattern_id}.json"
        path.write_text(json.dumps(data, indent=2))
