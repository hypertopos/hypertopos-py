# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lance as _lance  # type: ignore[import-untyped]  # noqa: E402
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hypertopos.model.sphere import (  # noqa: E402
    Alias,
    AliasFilter,
    ColumnSchema,
    CuttingPlane,
    DerivedPattern,
    EventDimDef,
    GMMComponent,
    GroupStats,
    LayerStorage,
    Line,
    PartitionConfig,
    Pattern,
    RelationDef,
    Sphere,
    StorageConfig,
)

_RECOVERABLE_READ_ERRORS = (
    OSError,
    ValueError,
    IndexError,
    KeyError,
    pa.ArrowInvalid,
    pa.ArrowTypeError,
)

_SEMANTICS_LIMITS: dict[str, int] = {
    "description": 200,
    "display_name": 60,
    "interpretation": 200,
}

# Maximum number of (pattern_id, version, primary_key) → row_id entries to cache.
_ROW_ID_CACHE_MAXSIZE = 2000

# Row threshold above which callers should prefer batched read paths to
# avoid loading the full dataset into memory in a single allocation.
BATCH_SCAN_THRESHOLD = 10_000_000  # rows; use batched path above this


class GDSReader:
    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._storage_config: StorageConfig | None = None
        self._points_cache: dict[tuple[str, int], pa.Table] = {}
        self._lance_dataset_cache: dict[tuple[str, int], Any] = {}
        # Lance version pinned at session open time for MVCC isolation.
        # Key: pattern_id (geometry) or "temporal:<pattern_id>" (temporal).
        # Value: Lance dataset version integer captured when the session opened.
        self._pinned_lance_versions: dict[str, int] = {}
        # LRU row-ID cache — maps (pattern_id, version, primary_key) to
        # the Lance row ID, enabling O(1) ds.take() on repeated reads of the
        # same entity.  Bypassed when MVCC pinning is active (pinned version row
        # IDs may not match row IDs in the current snapshot).
        self._row_id_cache: OrderedDict[tuple[str, int, str], int] = OrderedDict()
        # Tracks which points Lance datasets already have a BTREE index on primary_key.
        # Used by read_points_batch to lazily build the index on pre-existing spheres.
        self._points_btree_built: set[str] = set()
        # Cache of row IDs discovered by per-key BTREE equality scans.
        # Key: (line_id, version, primary_key) → Lance internal row ID.
        self._points_row_id_cache: dict[tuple, int] = {}

    def read_sphere(self) -> Sphere:
        raw = json.loads((self._base / "_gds_meta" / "sphere.json").read_text())
        lines = {k: self._parse_line(v) for k, v in raw.get("lines", {}).items()}
        patterns = {k: self._parse_pattern(v) for k, v in raw.get("patterns", {}).items()}
        aliases = {k: self._parse_alias(v) for k, v in raw.get("aliases", {}).items()}
        storage = self._parse_storage_config(raw)
        self._storage_config = storage
        sphere = Sphere(
            sphere_id=raw["sphere_id"],
            name=raw["name"],
            base_path=str(self._base),
            lines=lines,
            patterns=patterns,
            aliases=aliases,
            storage=storage,
        )
        for alias in sphere.aliases.values():
            cp = alias.filter.cutting_plane
            if cp is not None:
                pattern = sphere.patterns.get(alias.base_pattern_id)
                if pattern is not None and len(cp.normal) != pattern.delta_dim():
                    raise ValueError(
                        f"Alias '{alias.alias_id}': cutting_plane.normal has "
                        f"length {len(cp.normal)} but pattern "
                        f"'{alias.base_pattern_id}' has "
                        f"delta_dim={pattern.delta_dim()}"
                    )
        self._overlay_semantics(sphere)
        return sphere

    def _parse_storage_config(self, raw: dict[str, Any]) -> StorageConfig:
        storage_raw = raw.get("storage", {})
        cfg = StorageConfig()
        for layer_name in ("points", "geometry", "temporal", "invalidation_log", "forecast"):
            if layer_name in storage_raw:
                layer_raw = storage_raw[layer_name]
                layer = LayerStorage(
                    format=layer_raw.get("format", "lance"),
                    options=layer_raw.get("options", {}),
                )
                setattr(cfg, layer_name, layer)
        return cfg

    def _parse_line(self, raw: dict[str, Any]) -> Line:
        columns_raw = raw.get("columns")
        columns = (
            [ColumnSchema(name=c["name"], type=c["type"]) for c in columns_raw]
            if columns_raw is not None
            else None
        )
        role = raw["line_role"]
        if role == "context":
            warnings.warn(
                "line_role='context' is deprecated — treated as 'event'. Rebuild sphere.",
                DeprecationWarning,
                stacklevel=2,
            )
            role = "event"
        return Line(
            line_id=raw["line_id"],
            entity_type=raw["entity_type"],
            line_role=role,
            pattern_id=raw["pattern_id"],
            partitioning=PartitionConfig(
                mode=raw["partitioning"]["mode"],
                columns=raw["partitioning"]["columns"],
            ),
            versions=raw["versions"],
            columns=columns,
            fts_columns=raw.get("fts_columns"),
            source_id=raw.get("source_id"),
        )

    def _parse_pattern(self, raw: dict[str, Any]) -> Pattern:
        # Parse group_stats if present
        group_stats: dict[str, GroupStats] | None = None
        if "group_stats" in raw and raw["group_stats"]:
            group_stats = {}
            for gid, gs in raw["group_stats"].items():
                group_stats[gid] = GroupStats(
                    mu=np.array(gs["mu"], dtype=np.float32),
                    sigma_diag=np.array(gs["sigma_diag"], dtype=np.float32),
                    theta=np.array(gs["theta"], dtype=np.float32),
                    population_size=gs["population_size"],
                )

        # Parse dimension_weights if present
        dim_weights: np.ndarray | None = None
        if "dimension_weights" in raw and raw["dimension_weights"] is not None:
            dim_weights = np.array(raw["dimension_weights"], dtype=np.float32)

        event_dimensions = [
            EventDimDef(
                column=ed["column"],
                edge_max=float(ed["edge_max"]),
                display_name=ed.get("display_name"),
            )
            for ed in raw.get("event_dimensions", [])
        ]

        return Pattern(
            pattern_id=raw["pattern_id"],
            entity_type=raw["entity_type"],
            pattern_type=raw["pattern_type"],
            relations=[
                RelationDef(
                    line_id=r["line_id"],
                    direction=r["direction"],
                    required=r["required"],
                )
                for r in raw["relations"]
            ],
            mu=np.array(raw["mu"], dtype=np.float32),
            sigma_diag=np.array(raw["sigma_diag"], dtype=np.float32),
            theta=np.array(raw["theta"], dtype=np.float32),
            population_size=raw["population_size"],
            computed_at=datetime.fromisoformat(raw["computed_at"]),
            version=raw["version"],
            status=raw["status"],
            edge_max=(
                np.array(raw["edge_max"], dtype=np.float32)
                if raw.get("edge_max") is not None
                else None
            ),
            last_calibrated_at=(
                datetime.fromisoformat(raw["last_calibrated_at"])
                if raw.get("last_calibrated_at") else None
            ),
            prop_columns=raw.get("prop_columns", []),
            excluded_properties=raw.get("excluded_properties", []),
            group_stats=group_stats,
            group_by_property=raw.get("group_by_property"),
            dimension_weights=dim_weights,
            cholesky_inv=(
                np.array(raw["cholesky_inv"], dtype=np.float32)
                if "cholesky_inv" in raw and raw["cholesky_inv"] is not None
                else None
            ),
            gmm_components=(
                [
                    GMMComponent(
                        mu=np.array(gc["mu"], dtype=np.float32),
                        sigma_diag=np.array(gc["sigma_diag"], dtype=np.float32),
                        theta=np.array(gc["theta"], dtype=np.float32),
                        population_size=gc["population_size"],
                    )
                    for gc in raw["gmm_components"]
                ]
                if "gmm_components" in raw and raw["gmm_components"]
                else None
            ),
            event_dimensions=event_dimensions,
            entity_line_id=raw.get("entity_line"),
            dim_percentiles=raw.get("dim_percentiles"),
            timestamp_col=raw.get("timestamp_col"),
        )

    def _parse_alias(self, raw: dict[str, Any]) -> Alias:
        dp = raw["derived_pattern"]
        f = raw["filter"]
        cp_raw = f.get("cutting_plane")
        cutting_plane = (
            CuttingPlane(normal=cp_raw["normal"], bias=float(cp_raw["bias"]))
            if cp_raw
            else None
        )
        return Alias(
            alias_id=raw["alias_id"],
            base_pattern_id=raw["base_pattern_id"],
            filter=AliasFilter(
                include_relations=f["include_relations"],
                cutting_plane=cutting_plane,
            ),
            derived_pattern=DerivedPattern(
                mu=np.array(dp["mu"], dtype=np.float32),
                sigma_diag=np.array(dp["sigma_diag"], dtype=np.float32),
                theta=np.array(dp["theta"], dtype=np.float32),
                population_size=dp["population_size"],
                computed_at=datetime.fromisoformat(dp["computed_at"]),
            ),
            version=raw["version"],
            status=raw["status"],
        )

    def _overlay_semantics(self, sphere: Sphere) -> None:
        sem_path = self._base / "_gds_meta" / "semantics.json"
        if not sem_path.exists():
            return
        sem = json.loads(sem_path.read_text())

        if sphere_sem := sem.get("sphere"):
            desc = sphere_sem.get("description")
            if desc is not None:
                self._check_semantic_limit("sphere.description", desc, "description")
                sphere.description = desc

        for line_id, line_sem in sem.get("lines", {}).items():
            if line_id not in sphere.lines:
                continue
            desc = line_sem.get("description")
            if desc is not None:
                self._check_semantic_limit(f"lines.{line_id}.description", desc, "description")
                sphere.lines[line_id].description = desc

        for pat_id, pat_sem in sem.get("patterns", {}).items():
            if pat_id not in sphere.patterns:
                continue
            pat = sphere.patterns[pat_id]

            desc = pat_sem.get("description")
            if desc is not None:
                self._check_semantic_limit(f"patterns.{pat_id}.description", desc, "description")
                pat.description = desc

            rel_sems = pat_sem.get("relations", {})
            for rel in pat.relations:
                if rel.line_id not in rel_sems:
                    continue
                rel_sem = rel_sems[rel.line_id]

                dn = rel_sem.get("display_name")
                if dn is not None:
                    self._check_semantic_limit(
                        f"patterns.{pat_id}.relations.{rel.line_id}.display_name",
                        dn, "display_name"
                    )
                    rel.display_name = dn

                interp = rel_sem.get("interpretation")
                if interp is not None:
                    self._check_semantic_limit(
                        f"patterns.{pat_id}.relations.{rel.line_id}.interpretation",
                        interp, "interpretation"
                    )
                    rel.interpretation = interp

    def _check_semantic_limit(self, path: str, value: str, field: str) -> None:
        limit = _SEMANTICS_LIMITS[field]
        if len(value) > limit:
            raise ValueError(
                f"semantics.json: {path} exceeds {limit} chars "
                f"(got {len(value)})"
            )

    def read_geometry_stats(self, pattern_id: str, version: int) -> dict | None:
        """Read persisted geometry stats cache for a pattern version.

        Returns None if the cache file does not exist (caller should fall back
        to a full geometry scan).
        """
        path = (
            self._base
            / "_gds_meta"
            / "geometry_stats"
            / f"{pattern_id}_v{version}.json"
        )
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def read_population_forecast(self, pattern_id: str) -> pa.Table | None:
        """Read population forecast data for a pattern.

        Returns None if no forecast data exists.
        """
        lance_path = self._base / "forecast" / "population" / pattern_id / "data.lance"
        if not lance_path.exists():
            return None
        ds = _lance.dataset(str(lance_path))
        return ds.to_table()

    def geometry_column_names(
        self, pattern_id: str, version: int,
    ) -> set[str]:
        """Return the set of column names in a geometry Lance dataset."""
        folder = self._base / "geometry" / pattern_id / f"v={version}"
        lance_path = folder / "data.lance"
        if not lance_path.exists():
            return set()
        ds = _lance.dataset(str(lance_path))
        return {f.name for f in ds.schema}

    def read_geometry(
        self,
        pattern_id: str,
        version: int,
        primary_key: str | None = None,
        filters: dict[str, Any] | None = None,
        point_keys: list[str] | None = None,
        columns: list[str] | None = None,
        filter: str | None = None,
        sample_size: int | None = None,
    ) -> pa.Table:
        folder = self._base / "geometry" / pattern_id / f"v={version}"
        lance_path = str(folder / "data.lance")
        # Use session-pinned Lance version for MVCC isolation.
        pinned = self._pinned_lance_versions.get(pattern_id)
        if (
            sample_size is not None
            and primary_key is None
            and filter is None
            and filters is None
            and point_keys is None
        ):
            ds = _lance.dataset(lance_path)
            if pinned is not None:
                ds = _lance.dataset(lance_path, version=pinned)
            if sample_size >= ds.count_rows():
                return ds.to_table(columns=columns)
            table = ds.sample(sample_size, columns=columns, randomize_order=False)
            # Schema validation not applied to sampled reads (partial data)
            return table
        if point_keys is not None:
            table = self._read_lance_geometry_filtered(
                lance_path, point_keys, columns=columns, lance_version=pinned,
            )
        else:
            pk_filter: str | None = None
            if primary_key is not None:
                escaped_pk = primary_key.replace("'", "''")
                pk_filter = f"primary_key = '{escaped_pk}'"
            combined = (
                f"({pk_filter}) AND ({filter})" if pk_filter and filter
                else pk_filter or filter
            )
            # Pass cache_key only for pure single-key reads with no extra
            # filter and no MVCC pinning (pinned row IDs may differ across versions).
            row_id_cache_key: tuple[str, int, str] | None = (
                (pattern_id, version, primary_key)
                if primary_key is not None and filter is None and not self._pinned_lance_versions
                else None
            )
            table = self._read_lance(
                lance_path,
                columns=columns,
                filter=combined,
                lance_version=pinned,
                cache_key=row_id_cache_key,
            )
        # Fallback: vectorized filter for legacy geometry without entity_keys
        if (
            point_keys is not None
            and "entity_keys" not in table.schema.names
            and "edges" in table.schema.names
        ):
            table = self._filter_by_point_keys(table, point_keys)
        # Schema validation: full reads must contain all canonical columns
        if columns is None and table.num_rows > 0:
            from hypertopos.storage._schemas import GEOMETRY_REQUIRED_COLUMNS
            missing = GEOMETRY_REQUIRED_COLUMNS - set(table.schema.names)
            if missing:
                raise RuntimeError(
                    f"Geometry dataset for {pattern_id} v{version} is missing "
                    f"columns: {sorted(missing)}. Rebuild the sphere."
                )
        return table

    def count_geometry_rows(
        self, pattern_id: str, version: int, filter: str | None = None
    ) -> int:
        folder = self._base / "geometry" / pattern_id / f"v={version}"
        cache_key = (pattern_id, version)
        if cache_key not in self._lance_dataset_cache:
            self._lance_dataset_cache[cache_key] = _lance.dataset(
                str(folder / "data.lance")
            )
        return self._lance_dataset_cache[cache_key].count_rows(filter=filter)

    def read_geometry_batched(
        self,
        pattern_id: str,
        version: int,
        columns: list[str] | None = None,
        filter_expr: str | None = None,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Streaming read of geometry Lance dataset.

        Yields pa.RecordBatch chunks instead of loading the full table.
        Use for bulk scans on large spheres (>BATCH_SCAN_THRESHOLD rows).
        """
        lance_path = self._base / "geometry" / pattern_id / f"v={version}" / "data.lance"
        pinned = self._pinned_lance_versions.get(pattern_id)
        ds = _lance.dataset(str(lance_path))
        if pinned is not None and pinned != ds.latest_version:
            ds = _lance.dataset(str(lance_path), version=pinned)
        scanner = ds.scanner(
            filter=filter_expr,
            columns=columns,
            batch_size=batch_size,
        )
        yield from scanner.to_batches()

    def _read_lance_geometry_filtered(
        self,
        lance_path: str,
        point_keys: list[str],
        columns: list[str] | None = None,
        lance_version: int | None = None,
    ) -> pa.Table:
        """Read geometry from Lance with entity_keys pushdown filter.

        Falls back to full scan + vectorized filter when entity_keys column
        is missing (legacy geometry).
        """
        # Open the exact pinned version when supplied (MVCC isolation).
        # Optimization: skip version= when pinned equals latest.
        ds = _lance.dataset(lance_path)
        if lance_version is not None and lance_version != ds.latest_version:
            ds = _lance.dataset(lance_path, version=lance_version)
        # Check if entity_keys column exists in schema
        field_names = [f.name for f in ds.schema]
        schema_names = set(field_names)
        if columns is not None:
            columns = [c for c in columns if c in schema_names] or None
        if "entity_keys" not in field_names:
            # Fallback: full scan (caller will apply _filter_by_point_keys)
            return ds.to_table(columns=columns)
        # Build filter that matches on primary_key OR entity_keys.
        # primary_key match: needed when caller passes geometry row PKs
        #   (e.g. anomaly listing, chain lookup).
        # entity_keys match: needed when caller passes anchor entity keys
        #   to find event polygons referencing them.
        escaped = [k.replace("'", "''") for k in point_keys]
        pk_in = "primary_key IN ('" + "', '".join(escaped) + "')"
        if len(point_keys) == 1:
            ek_filter = f"array_contains(entity_keys, '{escaped[0]}')"
        else:
            keys_str = ", ".join(f"'{k}'" for k in escaped)
            ek_filter = f"array_has_any(entity_keys, [{keys_str}])"
        filter_expr = f"{pk_in} OR {ek_filter}"
        # entity_keys must be in scanner columns for pushdown to work
        scan_cols = columns
        if scan_cols is not None and "entity_keys" not in scan_cols:
            scan_cols = list(scan_cols) + ["entity_keys"]
        table = ds.scanner(filter=filter_expr, columns=scan_cols).to_table()
        if (
            columns is not None
            and "entity_keys" not in columns
            and "entity_keys" in table.schema.names
        ):
            table = table.drop("entity_keys")
        return table

    def _filter_by_point_keys(
        self, table: pa.Table, point_keys: list[str]
    ) -> pa.Table:
        """Filter geometry rows to those with at least one edge.point_key in point_keys.

        Uses vectorized pc.is_in on flattened struct array — O(n) Arrow C++.
        """
        key_arr = pa.array(point_keys, type=pa.string())
        edges_col = table["edges"].combine_chunks()
        flat_edges = pc.list_flatten(edges_col)
        flat_pkeys = pc.struct_field(flat_edges, "point_key")
        flat_match = pc.is_in(flat_pkeys, value_set=key_arr)

        # Map flat matches back to row indices via list offsets
        offsets_np = edges_col.offsets.to_numpy()
        matched_flat = np.where(flat_match.to_numpy(zero_copy_only=False))[0]
        row_indices = np.searchsorted(offsets_np[1:], matched_flat, side="right")
        keep = sorted(set(row_indices.tolist()))
        if not keep:
            return table.slice(0, 0)
        return table.take(keep)

    def read_temporal(
        self,
        pattern_id: str,
        primary_key: str,
        years: list[int] | None = None,
        from_slice: int | None = None,
        agent_id: str | None = None,
        filters: dict[str, str | list[str]] | None = None,
    ) -> pa.Table:
        lance_tables: list[pa.Table] = []
        lance_paths: list[Path] = [self._base / "temporal" / pattern_id / "data.lance"]
        if agent_id is not None:
            lance_paths.append(
                self._base / "temporal" / "_agents" / agent_id / pattern_id / "data.lance"
            )
        for lp in lance_paths:
            if lp.exists():
                lance_tables.append(
                    self._read_lance_temporal(lp, primary_key, filters)
                )
        if not lance_tables:
            return pa.table({})
        table = pa.concat_tables(lance_tables) if len(lance_tables) > 1 else lance_tables[0]
        if from_slice is not None and table.num_rows > 0:
            mask = pc.greater_equal(table["slice_index"], from_slice)
            table = table.filter(mask)
        return table

    def read_temporal_batch(
        self,
        pattern_id: str,
        filters: dict[str, str | list[str]] | None = None,
    ) -> pa.Table:
        """Read all temporal slices for a pattern in a single I/O pass.

        Unlike read_temporal(), does not filter by primary_key — returns rows
        for all entities. Use for population-wide scans such as π9_attract_drift.
        """
        lance_path = self._base / "temporal" / pattern_id / "data.lance"
        if not lance_path.exists():
            return pa.table({})
        # Use session-pinned Lance version for MVCC isolation.
        # Optimization: skip version= when pinned equals latest.
        pinned = self._pinned_lance_versions.get(f"temporal:{pattern_id}")
        ds = _lance.dataset(str(lance_path))
        if pinned is not None and pinned != ds.latest_version:
            ds = _lance.dataset(str(lance_path), version=pinned)
        table = ds.to_table()
        if filters and table.num_rows > 0:
            table = self._apply_temporal_filters(table, filters)
        return table

    def read_temporal_batched(
        self,
        pattern_id: str,
        batch_size: int = 65_536,
        timestamp_from: str | None = None,
        timestamp_to: str | None = None,
        keys: list[str] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Streaming read of temporal slices for a pattern.

        Yields pa.RecordBatch chunks. Passes timestamp_from/timestamp_to as
        Lance predicate pushdown — only relevant rows are read from disk.

        timestamp_from / timestamp_to: optional ISO-8601 strings (e.g. "2024-01-01"),
        half-open range [from, to). Naive timestamps treated as UTC.
        Partition pruning is automatic — the reader derives year/month hints
        from timestamp_from/timestamp_to by inspecting the directory structure, so agents
        do not need to pass year/month keys explicitly.
        For sub-year precision (month, quarter, specific date range), use ISO-8601 bounds
        (half-open range: from inclusive, to exclusive):
          timestamp_from="2024-06-01", timestamp_to="2024-10-01"

        keys: optional list of primary_key values to read. When provided, only rows
        for those specific entities are returned — avoids a full table scan when the
        caller has already sampled a subset of the population (e.g. π9 with sample_size).
        """
        lance_path = self._base / "temporal" / pattern_id / "data.lance"
        if not lance_path.exists():
            return
        pinned = self._pinned_lance_versions.get(f"temporal:{pattern_id}")
        ds = _lance.dataset(str(lance_path))
        if pinned is not None and pinned != ds.latest_version:
            ds = _lance.dataset(str(lance_path), version=pinned)
        # Build int64 microsecond SQL filter to avoid Lance's TimestampTz Substrait crash
        # (TimestampTz literals are unsupported in Lance's Substrait layer; CAST to BIGINT works).
        filter_parts: list[str] = []
        if keys is not None:
            if not keys:
                # Empty key list → no rows can match; return empty iterator
                return iter([])
            escaped = [k.replace("'", "''") for k in keys]
            pk_in = "primary_key IN (" + ", ".join(f"'{k}'" for k in escaped) + ")"
            filter_parts.append(pk_in)
        if timestamp_from:
            from_us = int(
                datetime.fromisoformat(timestamp_from).replace(tzinfo=UTC).timestamp()
                * 1_000_000
            )
            filter_parts.append(f"CAST(timestamp AS BIGINT) >= {from_us}")
        if timestamp_to:
            to_us = int(
                datetime.fromisoformat(timestamp_to).replace(tzinfo=UTC).timestamp()
                * 1_000_000
            )
            filter_parts.append(f"CAST(timestamp AS BIGINT) < {to_us}")
        sql_filter: str | None = " AND ".join(filter_parts) if filter_parts else None
        yield from ds.scanner(filter=sql_filter, batch_size=batch_size).to_batches()

    def _apply_temporal_filters(
        self,
        table: pa.Table,
        filters: dict[str, str | list[str]],
    ) -> pa.Table:
        """Apply year / timestamp_from / timestamp_to filters to a temporal table."""
        ts_int = pc.cast(table["timestamp"], pa.int64())  # μs since epoch

        # Record-level year filter (safety net after partition pruning)
        if "year" in filters:
            year_vals = filters["year"]
            if isinstance(year_vals, str):
                year_vals = [year_vals]
            year_ints = [int(y) for y in year_vals]
            if year_ints:
                masks = []
                for yr in year_ints:
                    start_us = int(datetime(yr, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
                    end_us = int(datetime(yr + 1, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
                    masks.append(pc.and_(
                        pc.greater_equal(ts_int, start_us),
                        pc.less(ts_int, end_us),
                    ))
                year_mask = masks[0]
                for m in masks[1:]:
                    year_mask = pc.or_(year_mask, m)
                table = table.filter(year_mask)
                ts_int = pc.cast(table["timestamp"], pa.int64())

        # Generic timestamp bounds — any granularity (month, quarter, day, etc.)
        if "timestamp_from" in filters and table.num_rows > 0:
            from_us = int(
                datetime.fromisoformat(filters["timestamp_from"])
                .replace(tzinfo=UTC)
                .timestamp()
                * 1_000_000
            )
            table = table.filter(pc.greater_equal(ts_int, from_us))
            ts_int = pc.cast(table["timestamp"], pa.int64())
        if "timestamp_to" in filters and table.num_rows > 0:
            to_us = int(
                datetime.fromisoformat(filters["timestamp_to"])
                .replace(tzinfo=UTC)
                .timestamp()
                * 1_000_000
            )
            table = table.filter(pc.less(ts_int, to_us))
        return table

    def _read_lance_temporal(
        self,
        lance_path: Path,
        primary_key: str,
        filters: dict[str, str | list[str]] | None,
    ) -> pa.Table:
        """Read temporal slices from a Lance dataset, filtering by primary_key."""
        ds = _lance.dataset(str(lance_path))
        table = ds.scanner(filter=f"primary_key = '{primary_key}'").to_table()
        if not filters or table.num_rows == 0:
            return table
        return self._apply_temporal_filters(table, filters)

    def read_points_schema(
        self,
        line_id: str,
        version: int,
    ) -> pa.Schema:
        cache_key = (line_id, version)
        cached = self._points_cache.get(cache_key)
        if cached is not None:
            return cached.schema
        lance_path = self._base / "points" / line_id / f"v={version}" / "data.lance"
        ds = _lance.dataset(str(lance_path))
        return ds.schema

    def read_points(
        self,
        line_id: str,
        version: int,
        filters: dict[str, Any] | None = None,
        primary_key: str | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        cache_key = (line_id, version)

        # Try cache for unfiltered reads (enrichment hot path)
        if filters is None and columns is None:
            cached = self._points_cache.get(cache_key)
            if cached is not None:
                if primary_key is not None:
                    return cached.filter(
                        pc.equal(cached["primary_key"], primary_key)
                    )
                return cached

        # Column projection — project from cache if warm, else targeted read
        if columns is not None and filters is None and primary_key is None:
            cached = self._points_cache.get(cache_key)
            if cached is not None:
                return cached.select(columns)
            base = self._base / "points" / line_id / f"v={version}"
            lance_path = base / "data.lance"
            ds = _lance.dataset(str(lance_path))
            return ds.scanner(columns=columns).to_table()

        base = self._base / "points" / line_id / f"v={version}"
        lance_path = base / "data.lance"

        if filters is not None:
            return self._read_lance_points(lance_path, primary_key, filters)

        # Cold cache + single-key read — targeted scan avoids loading
        # the full table into memory. Cache is NOT populated (single-key reads
        # are navigation hot-path, not enrichment; caching would waste memory).
        if primary_key is not None:
            return self._read_lance_points(lance_path, primary_key, None)

        base_table = self._read_lance_points(lance_path, None, None)

        # Cache unfiltered full table
        self._points_cache[cache_key] = base_table

        return base_table

    def search_points_fts(
        self,
        line_id: str,
        version: int,
        query: str,
        limit: int = 20,
    ) -> pa.Table:
        """Full-text search across all INVERTED-indexed string columns for a line.

        Requires INVERTED indices to be built at write time (GDSBuilder does this
        automatically). Returns a pa.Table with all point columns plus a '_score'
        column (BM25 relevance score). Rows are ordered by relevance descending.

        Raises ValueError if the Lance dataset does not exist for this line/version.
        """
        lance_path = self._base / "points" / line_id / f"v={version}" / "data.lance"
        if not lance_path.exists():
            raise ValueError(
                f"FTS requires Lance points for line '{line_id}' v={version}. "
                f"Dataset not found at {lance_path}"
            )
        ds = _lance.dataset(str(lance_path))
        return ds.scanner(full_text_query=query, limit=limit).to_table()

    def read_points_batch(
        self,
        line_id: str,
        version: int,
        primary_keys: list[str],
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Read specific rows by primary_key list. Efficient batch fetch.

        columns: optional column projection — read only these columns.
        Always includes primary_key even if not listed.

        If the full table is already cached, filters from it in-memory (O(n)).
        Otherwise uses per-key BTREE equality scans — O(log n) per key.
        For large key sets (>100), uses a single IN-filter scan instead of
        per-key loops for better throughput.
        Row IDs are cached after the first scan so subsequent calls for the same
        key are O(1) via ds.take().
        """
        if not primary_keys:
            base = self._base / "points" / line_id / f"v={version}"
            lance_path = base / "data.lance"
            ds = _lance.dataset(str(lance_path))
            schema = ds.schema
            return pa.table({f.name: pa.array([], type=f.type) for f in schema})

        def _project(table: pa.Table) -> pa.Table:
            if columns is None:
                return table
            cols = list(dict.fromkeys(["primary_key"] + columns))
            available = [c for c in cols if c in table.column_names]
            return table.select(available)

        cache_key = (line_id, version)
        cached = self._points_cache.get(cache_key)
        if cached is not None:
            mask = pc.is_in(cached["primary_key"], pa.array(primary_keys, type=pa.string()))
            return _project(cached.filter(mask))

        lance_path = self._base / "points" / line_id / f"v={version}" / "data.lance"
        if not lance_path.exists():
            return pa.table({})

        # Fast path: batch IN-filter for large key sets (>100 keys)
        if len(primary_keys) > 100:
            ds = _lance.dataset(str(lance_path))
            escaped = [k.replace("'", "''") for k in primary_keys]
            pk_in = ", ".join(f"'{k}'" for k in escaped)
            proj_cols = None
            if columns:
                proj_cols = list(dict.fromkeys(["primary_key"] + columns))
            try:
                result = ds.scanner(
                    filter=f"primary_key IN ({pk_in})",
                    columns=proj_cols,
                ).to_table()
                return result
            except _RECOVERABLE_READ_ERRORS:
                pass  # fall through to per-key path

        ds = _lance.dataset(str(lance_path))

        # Lazy BTREE build for existing spheres that predate the write-time index
        btree_key = str(lance_path)
        if btree_key not in self._points_btree_built:
            try:
                has_btree = any(
                    getattr(idx, "index_type", "").lower() in ("btree",)
                    and "primary_key" in getattr(idx, "columns", [])
                    for idx in ds.describe_indices()
                )
                if not has_btree:
                    ds.create_scalar_index("primary_key", index_type="BTREE")
            except _RECOVERABLE_READ_ERRORS:
                pass
            self._points_btree_built.add(btree_key)

        # Per-key equality scans — each uses BTREE (O(log n)); cache row IDs for O(1) future access
        tables: list[pa.Table] = []
        for key in primary_keys:
            row_cache_key = (line_id, version, key)
            if row_cache_key in self._points_row_id_cache:
                try:
                    row = ds.take([self._points_row_id_cache[row_cache_key]])
                    if row.num_rows == 1 and row["primary_key"][0].as_py() == key:
                        if "_rowid" in row.schema.names:
                            row = row.drop(["_rowid"])
                        tables.append(row)
                        continue
                except _RECOVERABLE_READ_ERRORS:
                    pass
                del self._points_row_id_cache[row_cache_key]
            escaped = key.replace("'", "''")
            try:
                result = ds.scanner(
                    filter=f"primary_key = '{escaped}'", with_row_id=True
                ).to_table()
            except _RECOVERABLE_READ_ERRORS:
                # Fallback: without row_id (older Lance versions)
                escaped2 = key.replace("'", "''")
                result = ds.scanner(filter=f"primary_key = '{escaped2}'").to_table()
                if result.num_rows > 0:
                    tables.append(result)
                continue
            if result.num_rows > 0:
                if "_rowid" in result.schema.names:
                    row_id = result["_rowid"][0].as_py()
                    self._points_row_id_cache[row_cache_key] = row_id
                    tables.append(result.drop(["_rowid"]))
                else:
                    tables.append(result)

        if not tables:
            schema = ds.schema
            return pa.table({f.name: pa.array([], type=f.type) for f in schema})
        return _project(pa.concat_tables(tables))

    def has_fts_index(self, line_id: str, version: int) -> bool:
        """Return True if the line's Lance dataset has at least one INVERTED (FTS) index."""
        lance_path = self._base / "points" / line_id / f"v={version}" / "data.lance"
        if not lance_path.exists():
            return False
        try:
            ds = _lance.dataset(str(lance_path))
            return any(
                getattr(idx, "index_type", "") == "Inverted"
                for idx in ds.describe_indices()
            )
        except _RECOVERABLE_READ_ERRORS:
            return False

    def _read_lance_points(
        self,
        lance_path: Path,
        primary_key: str | None,
        filters: dict[str, Any] | None,
    ) -> pa.Table:
        """Read points from Lance dataset. Uses scanner for predicate pushdown."""
        ds = _lance.dataset(str(lance_path))
        filter_parts: list[str] = []
        if primary_key is not None:
            filter_parts.append(f"primary_key = '{primary_key}'")
        if filters:
            for col, val in filters.items():
                if isinstance(val, str):
                    filter_parts.append(f"{col} = '{val}'")
                else:
                    filter_parts.append(f"{col} = {val}")
        if filter_parts:
            return ds.scanner(filter=" AND ".join(filter_parts)).to_table()
        return ds.to_table()

    def _read_files(self, paths: list[str]) -> pa.Table:
        if not paths:
            return pa.table({})
        tables = [pq.ParquetFile(path).read() for path in paths]
        if len(tables) == 1:
            return tables[0]
        return pa.concat_tables(tables)

    def _read_lance(
        self,
        lance_path: str,
        columns: list[str] | None = None,
        filter: str | None = None,
        lance_version: int | None = None,
        cache_key: tuple[str, int, str] | None = None,
    ) -> pa.Table:
        # Open the exact pinned version when supplied (MVCC isolation).
        # Optimization: skip version= when pinned equals latest (common case,
        # avoids Lance version reconstruction overhead — up to 12x faster).
        ds = _lance.dataset(lance_path)
        if lance_version is not None and lance_version != ds.latest_version:
            ds = _lance.dataset(lance_path, version=lance_version)
        if columns is not None:
            schema_names = {f.name for f in ds.schema}
            columns = [c for c in columns if c in schema_names] or None

        # Row-ID cache fast path — only when cache_key provided and not pinned.
        if cache_key is not None and cache_key in self._row_id_cache:
            try:
                row = ds.take([self._row_id_cache[cache_key]])
            except _RECOVERABLE_READ_ERRORS:
                # Row ID is out of range (e.g. after Lance compact/optimize) — evict.
                del self._row_id_cache[cache_key]
            else:
                if "_rowid" in row.schema.names:
                    row = row.drop(["_rowid"])
                # Verify row ID is still valid (row IDs change after compact/optimize).
                if row.num_rows == 1 and row["primary_key"][0].as_py() == cache_key[2]:
                    if columns is not None:
                        present = [c for c in columns if c in row.schema.names]
                        row = row.select(present) if present else row
                    return row
                # Stale cache entry — evict and fall through to scanner.
                del self._row_id_cache[cache_key]

        if filter is not None:
            if cache_key is not None:
                # Request _rowid alongside regular columns to populate cache.
                result = ds.scanner(
                    filter=filter, columns=columns, with_row_id=True
                ).to_table()
                if result.num_rows == 1:
                    rowid = int(result["_rowid"][0].as_py())
                    self._row_id_cache[cache_key] = rowid
                    if len(self._row_id_cache) > _ROW_ID_CACHE_MAXSIZE:
                        self._row_id_cache.popitem(last=False)  # evict oldest (LRU)
                if "_rowid" in result.schema.names:
                    result = result.drop(["_rowid"])
                return result
            return ds.scanner(filter=filter, columns=columns).to_table()
        return ds.to_table(columns=columns)

    def find_nearest_lance(
        self,
        pattern_id: str,
        version: int,
        query_vector: np.ndarray,
        k: int,
        exclude_keys: set[str] | None = None,
        filter_expr: str | None = None,
    ) -> list[tuple[str, float]] | None:
        """ANN search via Lance IVF_FLAT index. Returns None if no Lance dataset exists.

        filter_expr: optional Lance SQL predicate applied at ANN time (e.g. 'is_anomaly = true').
        Combines ANN + scalar filter in a single kernel pass for maximum efficiency.
        """
        lance_path = self._base / "geometry" / pattern_id / f"v={version}" / "data.lance"
        if not lance_path.exists():
            return None
        exclude = exclude_keys or set()
        k_fetch = (k + len(exclude) + 5) * 2
        ds = _lance.dataset(str(lance_path))
        scanner_kwargs: dict[str, Any] = {
            "nearest": {"column": "delta", "q": query_vector.tolist(), "k": k_fetch},
            "columns": ["primary_key", "delta"],
        }
        if filter_expr:
            scanner_kwargs["filter"] = filter_expr
        result = ds.scanner(**scanner_kwargs).to_table()
        out: list[tuple[str, float]] = []
        q = query_vector.astype(np.float32)
        for i in range(result.num_rows):
            pk = result["primary_key"][i].as_py()
            if pk in exclude:
                continue
            d = np.array(result["delta"][i].as_py(), dtype=np.float32)
            out.append((pk, float(np.linalg.norm(d - q))))
            if len(out) >= k:
                break
        return out

    def find_nearest_trajectory(
        self,
        pattern_id: str,
        query_vector: np.ndarray,
        k: int,
        exclude_keys: set[str] | None = None,
    ) -> list[dict] | None:
        """ANN search over trajectory summary vectors.

        Returns None if the trajectory index does not exist for this pattern.
        Returns list of dicts with keys: primary_key, distance, displacement,
        num_slices, first_timestamp, last_timestamp. Sorted by distance ascending.
        """
        traj_path = self._base / "_gds_meta" / "trajectory" / f"{pattern_id}.lance"
        if not traj_path.exists():
            return None
        exclude = exclude_keys or set()
        k_fetch = (k + len(exclude) + 5) * 2
        ds = _lance.dataset(str(traj_path))
        result = ds.scanner(
            nearest={
                "column": "trajectory_vector",
                "q": query_vector.astype(np.float32).tolist(),
                "k": k_fetch,
            },
            columns=[
                "primary_key", "trajectory_vector", "displacement",
                "num_slices", "first_timestamp", "last_timestamp",
            ],
        ).to_table()
        q = query_vector.astype(np.float32)
        out: list[dict] = []
        for i in range(result.num_rows):
            pk = result["primary_key"][i].as_py()
            if pk in exclude:
                continue
            tv = np.array(result["trajectory_vector"][i].as_py(), dtype=np.float32)
            dist = float(np.linalg.norm(tv - q))
            out.append({
                "primary_key": pk,
                "distance": dist,
                "displacement": result["displacement"][i].as_py(),
                "num_slices": result["num_slices"][i].as_py(),
                "first_timestamp": result["first_timestamp"][i].as_py(),
                "last_timestamp": result["last_timestamp"][i].as_py(),
            })
            if len(out) >= k:
                break
        out.sort(key=lambda x: x["distance"])
        return out

    def resolve_primary_keys_by_edge(
        self,
        pattern_id: str,
        version: int,
        line_id: str | None,
        point_key: str,
    ) -> list[str] | None:
        """Resolve primary keys via geometry entity_keys LABEL_LIST index.

        Returns list of primary_keys whose entity_keys contain *point_key*.
        When *line_id* is given, further filters to rows where the entity_key
        at the relation index matching *line_id* equals *point_key*.

        Returns None if the geometry dataset doesn't exist or lacks
        entity_keys column (caller should fall back to full scan).
        Returns empty list if no matches found.
        """
        folder = self._base / "geometry" / pattern_id / f"v={version}"
        lance_path = str(folder / "data.lance")
        if not folder.exists():
            return None
        ds = _lance.dataset(lance_path)
        field_names = {f.name for f in ds.schema}
        if "entity_keys" not in field_names:
            return None  # No entity_keys column — caller should fall back
        # LABEL_LIST index: find rows where entity_keys contains point_key
        pk_escaped = point_key.replace("'", "''")
        filter_expr = f"array_contains(entity_keys, '{pk_escaped}')"
        if line_id is None:
            tbl = ds.scanner(
                filter=filter_expr, columns=["primary_key"],
            ).to_table()
            return tbl.column("primary_key").to_pylist()
        # Filter by specific line_id position in entity_keys
        sphere = self.read_sphere()
        pattern = sphere.patterns.get(pattern_id)
        if pattern is None:
            tbl = ds.scanner(
                filter=filter_expr, columns=["primary_key"],
            ).to_table()
            return tbl.column("primary_key").to_pylist()
        rel_idx = None
        for i, rel in enumerate(pattern.relations):
            if rel.line_id == line_id:
                rel_idx = i
                break
        if rel_idx is None:
            return []
        # Read matching rows and post-filter by positional entity_keys[rel_idx]
        tbl = ds.scanner(
            filter=filter_expr, columns=["primary_key", "entity_keys"],
        ).to_table()
        results = []
        for i in range(tbl.num_rows):
            ek = tbl["entity_keys"][i].as_py()
            if ek and rel_idx < len(ek) and ek[rel_idx] == point_key:
                results.append(tbl["primary_key"][i].as_py())
        return results

    def read_temporal_centroids(
        self, pattern_id: str,
    ) -> list[dict] | None:
        """Read pre-computed temporal centroids. Returns None if cache doesn't exist."""
        cache_path = (
            self._base / "_gds_meta" / "temporal_centroids" / f"{pattern_id}.lance"
        )
        if not cache_path.exists():
            return None
        ds = _lance.dataset(str(cache_path))
        table = ds.to_table()
        result = []
        for i in range(table.num_rows):
            result.append({
                "window_start": table["window_start"][i].as_py(),
                "window_end": table["window_end"][i].as_py(),
                "centroid": table["centroid"][i].as_py(),
                "entity_count": table["entity_count"][i].as_py(),
                "anomaly_rate": table["anomaly_rate"][i].as_py(),
            })
        return result

    def read_calibration_tracker(
        self, pattern_id: str,
    ) -> CalibrationTracker | None:  # noqa: F821
        """Read calibration tracker from _gds_meta/calibration/{pattern_id}.json."""
        from hypertopos.engine.calibration import RESERVOIR_K, CalibrationTracker

        path = self._base / "_gds_meta" / "calibration" / f"{pattern_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())

        # Backward compat: older trackers lack norm_reservoir fields
        raw_reservoir = data.get("norm_reservoir", [])
        norm_reservoir = np.empty(RESERVOIR_K, dtype=np.float32)
        k = min(len(raw_reservoir), RESERVOIR_K)
        if k > 0:
            norm_reservoir[:k] = np.array(raw_reservoir[:k], dtype=np.float32)
        norm_reservoir_count = data.get("norm_reservoir_count", k)

        return CalibrationTracker(
            calibrated_mu=np.array(data["calibrated_mu"], dtype=np.float32),
            calibrated_sigma=np.array(data["calibrated_sigma"], dtype=np.float32),
            calibrated_theta=np.array(data["calibrated_theta"], dtype=np.float32),
            calibrated_n=data["calibrated_n"],
            calibrated_at=datetime.fromisoformat(data["calibrated_at"]),
            running_n=data["running_n"],
            running_mean=np.array(data["running_mean"], dtype=np.float32),
            running_m2=np.array(data["running_m2"], dtype=np.float32),
            soft_threshold=data.get("soft_threshold", 0.05),
            hard_threshold=data.get("hard_threshold", 0.20),
            norm_reservoir=norm_reservoir,
            norm_reservoir_count=norm_reservoir_count,
        )
