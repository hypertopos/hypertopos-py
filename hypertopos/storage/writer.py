# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import warnings
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lance as _lance
import numpy as np
import pyarrow as pa

from hypertopos.model.objects import SolidSlice
from hypertopos.storage.exceptions import GDSStorageError

if TYPE_CHECKING:
    from hypertopos.engine.calibration import CalibrationTracker

logger = logging.getLogger(__name__)

# Lance write defaults — every hypertopos write_dataset call goes through
# _write_lance() so all datasets share one explicit file format. The pylance
# "stable" alias moves between releases (2.1 → 2.2 in pylance 12), so the
# format is pinned to 2.2 for reproducible on-disk spheres.
#
# Schema constraint enforced by GDSBuilder: pattern delta vectors must have
# width >= 1. A zero-width fixed-size-list column triggers a Lance write
# panic in format 2.1+ (lance-format/lance#5102 — divide-by-zero in
# DictionaryDataBlock::decode_helper). The builder rejects 0-dimension
# patterns up front; format 2.0 silently allowed them.
#
# max_rows_per_group=8192 reduces metadata overhead vs default 1024.
_LANCE_WRITE_DEFAULTS: dict[str, Any] = {
    "data_storage_version": "2.2",
    "max_rows_per_group": 8192,
}

# Lance trains the IVF KMeans on at least this many vectors per partition.
# Below num_partitions * this, KMeans runs on under-filled (in the limit,
# empty) clusters: the build still costs time but the resulting ANN index is
# degenerate and offers no recall benefit over the brute-force scan the query
# path already falls back to when no index is present.
_IVF_MIN_ROWS_PER_PARTITION = 256

# How long write_sphere_json keeps retrying a replace that another open
# handle blocks (Windows file-sharing semantics).
_REPLACE_RETRY_SECONDS = 1.0


def _ivf_num_partitions(n_rows: int) -> int:
    """IVF partition count for an n-row vector dataset.

    Single source of truth for the ``num_partitions`` passed to every
    trajectory ``create_index`` call, so the sub-threshold guard
    (:func:`_ivf_index_worthwhile`) is computed against the exact value the
    index would be built with.
    """
    return min(64, max(16, int(n_rows**0.5)))


def _ivf_index_worthwhile(n_rows: int) -> bool:
    """True when an n-row dataset has enough vectors to train a useful IVF index.

    KMeans needs ~``_IVF_MIN_ROWS_PER_PARTITION`` samples per partition; below
    ``num_partitions * _IVF_MIN_ROWS_PER_PARTITION`` the index is degenerate
    and adds only build cost. The query path falls back to a correct
    brute-force scan when the index is absent, so skipping is zero recall loss.
    """
    return n_rows >= _ivf_num_partitions(n_rows) * _IVF_MIN_ROWS_PER_PARTITION


def _write_lance(
    table: pa.Table | pa.RecordBatchReader,
    path: str,
    mode: str = "create",
    **kwargs: Any,
) -> _lance.LanceDataset:
    """Write an Arrow table (or a stream of record batches) to Lance with optimized defaults."""
    merged = {**_LANCE_WRITE_DEFAULTS, **kwargs}
    return _lance.write_dataset(table, path, mode=mode, **merged)


def write_sphere_json(meta_dir: Path, data: dict[str, Any]) -> None:
    """Replace ``meta_dir / "sphere.json"`` atomically (see ``write_json_atomic``)."""
    write_json_atomic(meta_dir / "sphere.json", data)


def write_json_atomic(path: Path, data: Any, indent: int | None = 2) -> None:
    """Replace ``path`` with ``data`` as JSON atomically.

    The JSON goes to a uniquely named temporary sibling first and then
    replaces the file, so a reader never sees a half-written file and two
    writers never share a temporary file. On Windows the replace fails while
    another handle has the file open (a session taking its snapshot), so it
    is retried for up to ``_REPLACE_RETRY_SECONDS``.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=indent))
        deadline = time.monotonic() + _REPLACE_RETRY_SECONDS
        while True:
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
    finally:
        tmp.unlink(missing_ok=True)


_TEMPORAL_SCHEMA = pa.schema(
    [
        pa.field("primary_key", pa.string()),
        pa.field("slice_index", pa.int32()),
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("deformation_type", pa.string()),
        pa.field("shape_snapshot", pa.list_(pa.float32())),
        pa.field("pattern_ver", pa.int32()),
        pa.field("changed_property", pa.string()),
        pa.field("changed_line_id", pa.string()),
    ]
)


class GDSWriter:
    def __init__(self, base_path: str, branch: str | None = None) -> None:
        self._base = Path(base_path)
        # Geometry rewrites (ranks, reindex) commit to this sphere branch's
        # head when set, to main otherwise.
        self._branch = branch

    def _dataset(self, path: Path) -> _lance.LanceDataset:
        from hypertopos.storage._lance_branches import open_ref

        return open_ref(path, self._branch)

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
                "primary_key": [primary_key],
                "slice_index": [solid_slice.slice_index],
                "timestamp": [solid_slice.timestamp],
                "deformation_type": [solid_slice.deformation_type],
                "shape_snapshot": [shape_snapshot.tolist()],
                "pattern_ver": [solid_slice.pattern_ver],
                "changed_property": [solid_slice.changed_property],
                "changed_line_id": [solid_slice.changed_line_id],
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
        ds.optimize.compact_files(target_rows_per_fragment=4_194_304)
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
            mode_center = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2)
            median_val = float(np.median(delta_norms))
            if mode_count / n > 0.25 and mode_center < median_val:
                threshold = float(bin_edges[mode_bin + 1])
                inactive_count = int(np.sum(delta_norms <= threshold))
                inactive_ratio = round(inactive_count / n, 4)

        stats["inactive_ratio"] = inactive_ratio
        stats["inactive_count"] = inactive_count

        path = self._base / "_gds_meta" / "geometry_stats" / f"{pattern_id}_v{version}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, stats, indent=None)

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
        self,
        pattern_id: str,
        version: int = 1,
        *,
        _ds: Any = None,
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
        lance_path = self._base / "geometry" / pattern_id / "data.lance"
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
            lance_path,
            n_rows=n_rows,
            list_size=list_size,
            ds=ds,
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
        lance_path = self._base / "geometry" / pattern_id / "data.lance"
        if not lance_path.exists():
            return False
        ds = self._dataset(lance_path)
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
            indices = ds.describe_indices()
            for idx in indices:
                # field_names is the list of column names this index covers
                if "delta" in idx.field_names:
                    has_vector_index = True
                    if idx.num_rows_indexed is not None:
                        indexed_rows = int(idx.num_rows_indexed)
                    break  # delta vector index found, no need to keep scanning

        unindexed = n_rows if not has_vector_index else max(0, n_rows - indexed_rows)

        if unindexed / n_rows < threshold:
            return False

        # Compact first to merge fragments, then rebuild the vector index
        ds.optimize.compact_files(target_rows_per_fragment=4_194_304)

        delta_field = ds.schema.field("delta")
        list_size: int | None = (
            delta_field.type.list_size if hasattr(delta_field.type, "list_size") else None
        )
        if list_size is None or n_rows < 256:
            # Cannot build IVF_FLAT without fixed-size list or sufficient rows
            return False

        num_partitions = min(64, max(16, int(n_rows**0.5)))
        try:
            ds = self._dataset(lance_path)
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

    def recompute_delta_rank_pct(self, pattern_id: str) -> None:
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
        self._rewrite_population_ranks(pattern_id, rank=True, conformal=False)

    def recompute_conformal_p(self, pattern_id: str) -> None:
        """Recompute conformal_p globally across all entities in the pattern geometry.

        The conformal p-value is a whole-population right-tail statistic
        (``p = fraction of the population with delta_norm >= this entity's``,
        so *lower = more anomalous*). Incremental appends write a batch-local
        value, so — exactly like ``recompute_delta_rank_pct`` — it must be
        recomputed against the full population after an append, or freshly
        ingested rows carry a p-value relative only to their ingest batch.
        Same O(N) cost profile and call sites as ``recompute_delta_rank_pct``.
        """
        self._rewrite_population_ranks(pattern_id, rank=False, conformal=True)

    def recompute_rank_stats(self, pattern_id: str) -> None:
        """Recompute the population-relative geometry stats — ``delta_rank_pct``
        and ``conformal_p`` — in a single pass.

        Both are O(N) whole-population statistics over ``delta_norm``. Computing
        them together reads the geometry once and issues one commit, halving the
        append-path recompute I/O versus calling ``recompute_delta_rank_pct`` and
        ``recompute_conformal_p`` back to back. Recomputes whichever of the two
        columns the geometry actually carries.
        """
        self._rewrite_population_ranks(pattern_id, rank=True, conformal=True)

    def _rewrite_population_ranks(self, pattern_id: str, *, rank: bool, conformal: bool) -> None:
        """Rewrite the population rank columns in place, one fragment at a time.

        Both columns are functions of a row's ``delta_norm`` and the sorted
        population, so each fragment's values are computed against the global
        sort and joined back on row id inside that fragment, then committed as
        one update. Join memory is bounded by fragment size (compaction caps
        fragments) instead of population size, and only indexes on the
        rewritten columns go stale. A key-joined upsert of the whole population
        exhausts the query memory pool on multi-million-row patterns and also
        drops the key-lookup index.

        Like any read-then-commit rewrite, rows appended by another writer
        between the read and the commit keep their batch-local values until
        the next recompute.
        """
        from hypertopos.builder._stats import compute_conformal_p, population_left_ranks

        lance_path = self._base / "geometry" / pattern_id / "data.lance"
        if not lance_path.exists():
            return
        ds = self._dataset(lance_path)
        field_names = {f.name for f in ds.schema}
        rank = rank and "delta_rank_pct" in field_names
        conformal = conformal and "conformal_p" in field_names
        if not (rank or conformal):
            return
        fragments = ds.get_fragments()
        parts = [f.to_table(columns=["delta_norm"], with_row_id=True) for f in fragments]
        if not parts:
            return
        delta_norms = np.concatenate(
            [p["delta_norm"].to_numpy(zero_copy_only=False) for p in parts]
        )
        n = len(delta_norms)
        if n == 0:
            return
        ranks = population_left_ranks(delta_norms)
        values: dict[str, np.ndarray] = {}
        if rank:
            values["delta_rank_pct"] = (ranks / n * 100).astype(np.float32)
        if conformal:
            values["conformal_p"] = compute_conformal_p(delta_norms, ranks)

        bounds = np.cumsum([0] + [p.num_rows for p in parts])
        updated: list[Any] = []
        modified: set[int] = set()
        for fragment, part, lo, hi in zip(fragments, parts, bounds[:-1], bounds[1:], strict=True):
            if hi == lo:
                continue
            columns: dict[str, Any] = {"_rowid": part["_rowid"]}
            for name, column in values.items():
                columns[name] = pa.array(column[lo:hi], type=pa.float32())
            # _rowid is the fragment-local row address here: geometry datasets
            # do not enable stable row ids.
            metadata, field_ids = fragment.update_columns(pa.table(columns), left_on="_rowid")
            updated.append(metadata)
            modified.update(field_ids)
        _lance.LanceDataset.commit(
            ds if self._branch else str(lance_path),
            _lance.LanceOperation.Update(
                updated_fragments=updated, fields_modified=sorted(modified)
            ),
            read_version=ds.version,
        )
        # The update drops the rewritten fragments from indexes on those
        # columns; fold them back in so filters on the ranks stay indexed.
        ds = self._dataset(lance_path)
        stale = [i.name for i in ds.describe_indices() if set(i.field_names) & values.keys()]
        if stale:
            ds.optimize.optimize_indices(index_names=stale)

    def replace_geometry_columns(
        self,
        lance_path: Path,
        branch: str | None,
        row_ids: list[pa.Array],
        values: dict[str, pa.Array],
    ) -> Any:
        """Replace whole columns of a geometry dataset in one commit; reindex only them.

        ``row_ids`` holds each fragment's ``_rowid`` array in ``get_fragments()``
        order and ``values`` full-length columns in the same row order — the
        order a per-fragment read with ``with_row_id=True`` returns. Columns
        not in ``values`` and their indexes (``primary_key``, ``entity_keys``)
        are not rewritten. Commits on ``branch``'s head when given, else on
        main. Returns the dataset at the new head.
        """
        from hypertopos.storage import _lance_branches

        ds = _lance_branches.open_ref(lance_path, branch)
        fragments = ds.get_fragments()
        if len(fragments) != len(row_ids):
            raise GDSStorageError(
                f"{lance_path}: {len(fragments)} fragments but row ids for {len(row_ids)}"
            )
        bounds = np.cumsum([0] + [len(r) for r in row_ids])
        updated: list[Any] = []
        modified: set[int] = set()
        for fragment, rid, lo, hi in zip(fragments, row_ids, bounds[:-1], bounds[1:], strict=True):
            if hi == lo:
                continue
            if int(rid[0].as_py()) >> 32 != fragment.fragment_id:
                raise GDSStorageError(f"{lance_path}: row ids do not match the fragments")
            columns: dict[str, Any] = {"_rowid": rid}
            for name, column in values.items():
                columns[name] = column.slice(int(lo), int(hi - lo))
            # _rowid is the fragment-local row address: geometry datasets do
            # not enable stable row ids.
            metadata, field_ids = fragment.update_columns(pa.table(columns), left_on="_rowid")
            updated.append(metadata)
            modified.update(field_ids)
        _lance.LanceDataset.commit(
            ds if branch else str(lance_path),
            _lance.LanceOperation.Update(
                updated_fragments=updated, fields_modified=sorted(modified)
            ),
            read_version=ds.version,
        )
        ds = _lance_branches.open_ref(lance_path, branch)
        n_rows = ds.count_rows()
        if "delta" in values and n_rows >= 256:
            ds.create_index(
                "delta",
                index_type="IVF_FLAT",
                num_partitions=min(64, max(16, int(n_rows**0.5))),
                replace=True,
            )
            ds = _lance_branches.open_ref(lance_path, branch)
        # The update drops the rewritten fragments from scalar indexes on those
        # columns; fold them back in so filters on them stay indexed.
        stale = [
            i.name
            for i in ds.describe_indices()
            if set(i.field_names) & values.keys() and i.field_names != ["delta"]
        ]
        if stale:
            ds.optimize.optimize_indices(index_names=stale)
            ds = _lance_branches.open_ref(lance_path, branch)
        return ds

    def append_geometry(self, table: pa.Table, pattern_id: str) -> None:
        """Append new geometry rows to an existing Lance geometry dataset.

        Writes the rows via lance write_dataset(mode='append') and then checks
        whether the ANN index needs rebuilding (more than 10% unindexed rows).

        The delta column must be a fixed-size list of float32 — the method casts
        variable-size lists automatically when num_rows > 0.

        Automatically recomputes delta_rank_pct globally after appending to ensure
        percentile ranks reflect the full population, not just the appended batch.
        """
        lance_path = self._base / "geometry" / pattern_id / "data.lance"
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

        if lance_path.exists():
            # Upsert: replace existing rows with same primary_key, insert new ones.
            # Prevents duplicate rows from accumulating after interrupted incremental_update.
            ds = _lance.dataset(str(lance_path))
            ds.merge_insert(
                "primary_key"
            ).when_matched_update_all().when_not_matched_insert_all().execute(table)
        else:
            _write_lance(table, str(lance_path), mode="create")
        self._maybe_reindex_geometry(pattern_id)
        self.recompute_rank_stats(pattern_id)

    def _build_indices_on_lance_path(
        self,
        lance_path: Path,
        n_rows: int,
        list_size: int | None,
        *,
        ds: Any = None,
        skip_vector_index: bool = False,
    ) -> None:
        """Build IVF_FLAT vector index and scalar indices on geometry Lance dataset."""
        if ds is None:
            ds = _lance.dataset(str(lance_path))

        # IVF_FLAT vector index on delta — O(N) build vs O(N×P×iters) for IVF_PQ.
        # 6M rows: IVF_FLAT builds in ~3s, IVF_PQ takes 60-120s.
        # Query: 23ms vs 14ms — negligible for agent workloads.
        if list_size is not None and n_rows >= 256 and not skip_vector_index:
            num_partitions = min(64, max(16, int(n_rows**0.5)))
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
        # delta_dim_* columns are deliberately left unindexed: Lance prunes
        # only through an explicit scalar index, so geometry_filters and
        # aggregation pushdown on a delta_dim_k predicate scan that column.
        # One BTREE per dimension would make those filters several times
        # faster, but costs an index build per dimension on every pattern
        # for an occasional, user-chosen filter. A ZONEMAP barely helps,
        # since rows are stored in key order, not by delta value.

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
        bucket_timestamps: list[Any],
        *,
        raw_shapes: np.ndarray,
    ) -> int:
        """Build trajectory vectors directly from an in-memory delta tensor.

        Eliminates the need to re-read temporal Lance data. Called by
        build_temporal() with the tensor already in memory and already
        converted to deltas in the pattern's metric.

        Only each entity's active windows count (the rule drift uses), judged
        on ``raw_shapes``; entities with fewer than two active windows get no
        trajectory.

        shape_tensor: (n_entities, n_buckets, D) float32 deltas
        primary_keys: list of entity keys, len == n_entities
        bucket_timestamps: list of datetime per bucket
        raw_shapes: (n_entities, n_buckets, D) shape snapshots behind the deltas
        """
        from hypertopos.engine.geometry import active_window_trajectories

        n, n_buckets, D = shape_tensor.shape
        if n == 0 or n_buckets < 2:
            return 0

        d2 = D * 2
        kept, traj_vectors, displacements, num_slices_arr, first_w, last_w = (
            active_window_trajectories(raw_shapes, shape_tensor)
        )
        n = len(kept)
        if n == 0:
            return 0
        primary_keys = [primary_keys[i] for i in kept]

        traj_vec_col = pa.FixedSizeListArray.from_arrays(
            pa.array(traj_vectors.ravel(), type=pa.float32()),
            list_size=d2,
        ).cast(pa.list_(pa.float32(), d2))

        traj_schema = pa.schema(
            [
                pa.field("primary_key", pa.string()),
                pa.field("trajectory_vector", pa.list_(pa.float32(), d2)),
                pa.field("displacement", pa.float32()),
                pa.field("num_slices", pa.int32()),
                pa.field("first_timestamp", pa.timestamp("us", tz="UTC")),
                pa.field("last_timestamp", pa.timestamp("us", tz="UTC")),
            ]
        )

        traj_table = pa.table(
            {
                "primary_key": pa.array(primary_keys, type=pa.string()),
                "trajectory_vector": traj_vec_col,
                "displacement": pa.array(displacements, type=pa.float32()),
                "num_slices": pa.array(num_slices_arr, type=pa.int32()),
                "first_timestamp": pa.array(
                    [bucket_timestamps[i] for i in first_w],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "last_timestamp": pa.array(
                    [bucket_timestamps[i] for i in last_w],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            },
            schema=traj_schema,
        )

        traj_path = self._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        _write_lance(traj_table, str(traj_path), mode="overwrite")

        # IVF_FLAT index. Below num_partitions * 256 rows the KMeans training is
        # starved and the index is degenerate; skip it so the build stays short
        # — trajectory search falls back to a correct brute-force scan.
        if _ivf_index_worthwhile(n):
            traj_ds = _lance.dataset(str(traj_path))
            n_partitions = _ivf_num_partitions(n)
            with suppress(Exception):
                traj_ds.create_index(
                    "trajectory_vector",
                    index_type="IVF_FLAT",
                    num_partitions=n_partitions,
                )
        elif n > 0:
            logger.info(
                "trajectory %s: %d rows below ANN training threshold "
                "(%d) — using full-scan fallback",
                pattern_id,
                n,
                _ivf_num_partitions(n) * _IVF_MIN_ROWS_PER_PARTITION,
            )

        return int(n)

    def build_trajectory_index(self, pattern_id: str) -> int:
        """Build trajectory summary vectors from temporal slices.

        Trajectory vector = concat([mean(deltas), std(deltas)]) over each
        entity's active windows (the rule drift uses; entities with fewer than
        two get no row). Captures both direction and volatility of geometric
        change. Shape snapshots are converted to deltas in the pattern's stored
        metric (as recorded in sphere.json), the same conversion the build uses.

        Returns the number of entities indexed, or 0 if no temporal data exists.
        """
        from hypertopos.engine.geometry import (
            active_window_rows,
            slice_deltas,
            trajectory_summary,
        )
        from hypertopos.storage.reader import GDSReader

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
                "Rebuild the sphere to upgrade first."
            )

        pattern = GDSReader(str(self._base)).read_sphere().patterns[pattern_id]

        # Read all temporal data at once — sort by (pk, timestamp) for grouped processing
        full_table = ds.to_table(
            columns=["primary_key", "shape_snapshot", "timestamp"],
        )
        if full_table.num_rows == 0:
            return 0

        # Sort by (primary_key, timestamp) — Arrow sort, no Python
        import pyarrow.compute as pc

        sort_idx = pc.sort_indices(
            full_table, sort_keys=[("primary_key", "ascending"), ("timestamp", "ascending")]
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
        D_shape = (
            shape_col.type.list_size
            if hasattr(shape_col.type, "list_size")
            else len(shape_col[0].as_py())
        )
        shape_matrix = (
            shape_col.values.to_numpy(zero_copy_only=False).reshape(-1, D_shape).astype(np.float32)
        )

        def _group_starts(keys: pa.Array) -> np.ndarray:
            boundary = np.ones(len(keys), dtype=bool)
            boundary[1:] = pc.not_equal(keys[1:], keys[:-1]).to_numpy(zero_copy_only=False)
            return np.flatnonzero(boundary)

        # Only active windows count (the rule drift uses); entities left with
        # fewer than two get no trajectory.
        group_starts = _group_starts(pk_arr)
        active = active_window_rows(shape_matrix, group_starts)
        n_active = np.add.reduceat(active.astype(np.int64), group_starts)
        counts_all = np.diff(np.append(group_starts, len(active)))
        active &= np.repeat(n_active >= 2, counts_all)
        if not active.any():
            return 0
        row_idx = np.flatnonzero(active)
        pk_arr = pk_arr.take(pa.array(row_idx))
        ts_arr = ts_arr.take(pa.array(row_idx))
        shape_matrix = slice_deltas(shape_matrix[row_idx], pattern)

        group_starts = _group_starts(pk_arr)
        group_ends = np.append(group_starts[1:], len(row_idx))
        n = len(group_starts)
        d2 = shape_matrix.shape[1] * 2
        traj_vectors, displacements, counts = trajectory_summary(shape_matrix, group_starts)
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

        traj_schema = pa.schema(
            [
                pa.field("primary_key", pa.string()),
                pa.field("trajectory_vector", pa.list_(pa.float32(), d2)),
                pa.field("displacement", pa.float32()),
                pa.field("num_slices", pa.int32()),
                pa.field("first_timestamp", pa.timestamp("us", tz="UTC")),
                pa.field("last_timestamp", pa.timestamp("us", tz="UTC")),
            ]
        )

        traj_table = pa.table(
            {
                "primary_key": pa.array(pks, type=pa.string()),
                "trajectory_vector": traj_vec_col,
                "displacement": pa.array(displacements, type=pa.float32()),
                "num_slices": pa.array(num_slices, type=pa.int32()),
                "first_timestamp": pa.array(first_ts, type=pa.timestamp("us", tz="UTC")),
                "last_timestamp": pa.array(last_ts, type=pa.timestamp("us", tz="UTC")),
            },
            schema=traj_schema,
        )

        traj_path = self._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        _write_lance(traj_table, str(traj_path), mode="overwrite")

        # IVF_FLAT index — O(N) build vs O(N×P×iters) for IVF_PQ.
        # 150K entities: IVF_FLAT builds in <1s, IVF_PQ takes 3-5 min.
        # Query latency is comparable (4ms vs 25ms flat scan at 150K).
        # Below num_partitions * 256 rows the KMeans training is starved and
        # the index is degenerate, so skip it — trajectory search falls back to
        # a correct brute-force scan (zero recall loss, pure build-time shrink).
        if _ivf_index_worthwhile(n):
            traj_ds = _lance.dataset(str(traj_path))
            n_partitions = _ivf_num_partitions(n)
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
        elif n > 0:
            logger.info(
                "trajectory %s: %d rows below ANN training threshold "
                "(%d) — using full-scan fallback",
                pattern_id,
                n,
                _ivf_num_partitions(n) * _IVF_MIN_ROWS_PER_PARTITION,
            )
        return n

    def write_temporal_centroids(
        self,
        pattern_id: str,
        centroids: list[dict[str, Any]],
    ) -> None:
        """Write pre-computed per-window centroids for pi11/pi12 fast path."""
        table = pa.table(
            {
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
            }
        )

        out_dir = self._base / "_gds_meta" / "temporal_centroids"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pattern_id}.lance"
        _write_lance(table, str(out_path), mode="overwrite")

    def invalidate_temporal_centroids(self, pattern_id: str, *, _ds: Any = None) -> None:
        """Empty the temporal centroid cache; an empty cache reads as absent.

        Rows are deleted rather than the dataset removed: sphere branches hold
        a Lance branch on this dataset. ``_ds`` is the handle to commit
        through (a branch head); by default the main dataset.
        """
        if _ds is not None:
            _ds.delete("true")
            return
        path = self._base / "_gds_meta" / "temporal_centroids" / f"{pattern_id}.lance"
        if path.exists():
            _lance.dataset(str(path)).delete("true")

    # ── Edge table ──────────────────────────────────────────────

    def write_edges(
        self,
        pattern_id: str,
        edges_table: pa.Table,
    ) -> None:
        """Write edge table as Lance dataset with BTREE indexes.

        Schema: from_key, to_key, event_key, timestamp, amount.
        Used for runtime graph traversal and lazy chain discovery.
        """
        lance_dir = self._base / "edges" / pattern_id
        lance_dir.mkdir(parents=True, exist_ok=True)
        lance_path = str(lance_dir / "data.lance")
        _write_lance(edges_table, lance_path, mode="overwrite")
        self._create_edge_indexes(lance_path)

    def append_edges(
        self,
        pattern_id: str,
        new_edges: pa.Table,
    ) -> None:
        """Append new edges to existing Lance dataset (streaming build)."""
        lance_path = str(self._base / "edges" / pattern_id / "data.lance")
        _write_lance(new_edges, lance_path, mode="append")

    def create_edge_indexes(self, pattern_id: str) -> None:
        """Build BTREE indexes on from_key and to_key after streaming writes."""
        lance_path = str(self._base / "edges" / pattern_id / "data.lance")
        self._create_edge_indexes(lance_path)

    def _create_edge_indexes(self, lance_path: str) -> None:
        ds = _lance.dataset(lance_path)
        if ds.count_rows() == 0:
            return
        with suppress(Exception):
            ds.create_scalar_index("from_key", index_type="BTREE", replace=True)
        with suppress(Exception):
            ds.create_scalar_index("to_key", index_type="BTREE", replace=True)

    def write_edge_stats(self, pattern_id: str, stats: dict[str, Any]) -> None:
        """Cache edge table statistics as JSON for fast reads."""
        path = self._base / "_gds_meta" / "edge_stats"
        path.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path / f"{pattern_id}.json", stats)

    def write_contagion_stats(
        self,
        pattern_id: str,
        table: pa.Table,
    ) -> None:
        """Persist precomputed per-entity contagion statistics.

        Schema (enforced by caller): primary_key (string), neighbor_count (int32),
        anomalous_neighbor_count (int32), contagion_ratio (float32). One row per
        entity that appears as either endpoint in the pattern's edge table. The
        scanner reads this table directly instead of replaying the full edge
        table on every passive_scan / composite_risk invocation.
        """
        cs_dir = self._base / "_gds_meta" / "contagion_stats"
        cs_dir.mkdir(parents=True, exist_ok=True)
        lance_path = str(cs_dir / f"{pattern_id}.lance")
        _write_lance(table, lance_path, mode="overwrite")
        ds = _lance.dataset(lance_path)
        if ds.count_rows() > 0:
            with suppress(Exception):
                ds.create_scalar_index(
                    "primary_key",
                    index_type="BTREE",
                    replace=True,
                )

    def write_calibration_tracker(
        self,
        pattern_id: str,
        tracker: CalibrationTracker,
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
        write_json_atomic(cal_dir / f"{pattern_id}.json", data)
