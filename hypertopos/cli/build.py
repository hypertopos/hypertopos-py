# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""YAML config → GDSBuilder bridge for ``hypertopos build``.

Translates a parsed ``SphereConfig`` into the correct sequence of
GDSBuilder calls: add_line, add_pattern, add_event_dimension,
add_derived_dimension, add_composite_line, build, build_temporal.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from hypertopos.cli.schema import (
    AliasConfig,
    ChainLineConfig,
    CompositeLineConfig,
    DerivedDimGroup,
    FeatureSpec,
    LineConfig,
    PatternConfig,
    SphereConfig,
    TemporalConfig,
    parse_config,
)
from hypertopos.cli.sources import load_source


def run_build(
    config_path: str,
    output_dir: str | None,
    force: bool,
    verbose: bool,
    no_temporal: bool = False,
    no_chains: bool = False,
    no_edges: bool = False,
) -> None:
    """Execute the ``hypertopos build`` command."""
    try:
        cfg = parse_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(config_path).resolve().parent

    # Determine output path
    out_path = Path(output_dir) if output_dir else base_dir / f"gds_{cfg.sphere_id}"

    if out_path.exists():
        if not force:
            print(
                f"error: output directory '{out_path}' already exists. "
                "Use --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)
        shutil.rmtree(out_path)

    try:
        _do_build(
            cfg, base_dir, str(out_path), verbose,
            no_temporal=no_temporal, no_chains=no_chains,
            no_edges=no_edges,
        )
        print(f"Built: {out_path}")
    except Exception as exc:
        print(f"error: build failed: {exc}", file=sys.stderr)
        sys.exit(1)


def run_validate(config_path: str) -> None:
    """Execute ``hypertopos validate`` — parse + check sources exist."""
    try:
        cfg = parse_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(config_path).resolve().parent
    errors: list[str] = []

    # Check all source files exist
    for name, src in cfg.sources.items():
        if src.path:
            p = Path(src.path)
            if not p.is_absolute():
                p = base_dir / p
            if not p.exists():
                errors.append(f"Source '{name}': file not found: {p}")
            # Check join files
            for j in src.join:
                jp = Path(j.file)
                if not jp.is_absolute():
                    jp = base_dir / jp
                if not jp.exists():
                    errors.append(
                        f"Source '{name}' join: file not found: {jp}"
                    )
        elif src.script:
            sp = Path(src.script)
            if not sp.is_absolute():
                sp = base_dir / sp
            if not sp.exists():
                errors.append(f"Source '{name}': script not found: {sp}")

    # Check cross-references
    for pid, pat in cfg.patterns.items():
        if pat.event_dimensions and pat.type != "event":
            errors.append(
                f"Pattern '{pid}': event_dimensions only valid "
                "on type=event patterns"
            )
        if pat.derived_dimensions:
            for group in pat.derived_dimensions:
                fp = group.from_pattern
                if fp and fp not in cfg.patterns and fp not in cfg.lines:
                    errors.append(
                        f"Pattern '{pid}' derived_dimensions: "
                        f"from_pattern '{group.from_pattern}' not found"
                    )

    for tc in cfg.temporal:
        if tc.event_line not in cfg.lines:
            errors.append(
                f"Temporal config for '{tc.pattern}': "
                f"event_line '{tc.event_line}' not in lines"
            )
        pat = cfg.patterns.get(tc.pattern)
        if pat and pat.type != "anchor":
            errors.append(
                f"Temporal config for '{tc.pattern}': "
                f"pattern must be anchor type, got '{pat.type}'"
            )

    # Check for duplicate dimension names within a pattern
    for pid, pat in cfg.patterns.items():
        dim_names: list[str] = []
        if pat.event_dimensions:
            dim_names.extend(ed.column for ed in pat.event_dimensions)
        if pat.precomputed_dimensions:
            dim_names.extend(pc.column for pc in pat.precomputed_dimensions)
        if pat.derived_dimensions:
            for group in pat.derived_dimensions:
                dim_names.extend(f.dimension_name for f in group.features)
        seen: set[str] = set()
        for dn in dim_names:
            if dn in seen:
                errors.append(
                    f"Pattern '{pid}': duplicate dimension name '{dn}'"
                )
            seen.add(dn)

    # Check for cycles in from_pattern references
    def _has_cycle(start: str, visited: set[str]) -> bool:
        if start in visited:
            return True
        visited.add(start)
        pat = cfg.patterns.get(start)
        if pat and pat.derived_dimensions:
            for group in pat.derived_dimensions:
                if (
                    group.from_pattern
                    and group.from_pattern in cfg.patterns
                    and _has_cycle(group.from_pattern, visited)
                ):
                    return True
        visited.discard(start)
        return False

    for pid in cfg.patterns:
        if _has_cycle(pid, set()):
            errors.append(
                f"Pattern '{pid}': cycle detected in from_pattern references"
            )

    if errors:
        print("Validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    n_src = len(cfg.sources)
    n_lines = len(cfg.lines)
    n_pat = len(cfg.patterns)
    n_cl = len(cfg.composite_lines)
    n_tmp = len(cfg.temporal)
    print(
        f"Valid: {cfg.sphere_id} "
        f"({n_src} sources, {n_lines} lines, {n_pat} patterns, "
        f"{n_cl} composite lines, {n_tmp} temporal configs)"
    )


def build_from_config(
    cfg: SphereConfig,
    output_path: Path,
    source_dir: Path | None = None,
    verbose: bool = False,
    no_temporal: bool = False,
    no_chains: bool = False,
) -> None:
    """Build a sphere from a parsed SphereConfig.

    Public API for programmatic builds (used by tests and external callers).
    ``source_dir`` defaults to current working directory if not specified.
    """
    base_dir = source_dir or Path.cwd()
    _do_build(
        cfg, base_dir, str(output_path), verbose,
        no_temporal=no_temporal, no_chains=no_chains,
    )


# ── internal build logic ─────────────────────────────────────────────


def _do_build(
    cfg: SphereConfig,
    base_dir: Path,
    output_path: str,
    verbose: bool,
    no_temporal: bool = False,
    no_chains: bool = False,
    no_edges: bool = False,
) -> None:
    """Core build: loads sources, wires GDSBuilder, calls build + temporal."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa

    from hypertopos.builder.builder import GDSBuilder

    t_total = time.time()
    builder = GDSBuilder(
        cfg.sphere_id, output_path,
        name=cfg.name, description=cfg.description,
    )

    # 1. Load all sources (parallel when >1)
    t_phase = time.time()
    tables: dict[str, pa.Table] = {}
    source_items = list(cfg.sources.items())
    if len(source_items) > 1:
        def _load_one(name_cfg: tuple[str, object]) -> tuple[str, pa.Table]:
            name, src_cfg = name_cfg
            return name, load_source(src_cfg, base_dir)

        with ThreadPoolExecutor(max_workers=min(4, len(source_items))) as pool:
            for name, table in pool.map(_load_one, source_items):
                tables[name] = table
                if verbose:
                    print(f"  Source '{name}': {table.num_rows:,} rows, "
                          f"{table.num_columns} cols")
    else:
        for name, src_cfg in source_items:
            tables[name] = load_source(src_cfg, base_dir)
            if verbose:
                t = tables[name]
                print(f"  Source '{name}': {t.num_rows:,} rows, "
                      f"{t.num_columns} cols")
    if verbose:
        print(f"  Sources total: {time.time() - t_phase:.1f}s")

    # 1b. Early column validation — fail fast before heavy work
    col_errors: list[str] = []
    for line_id, line_cfg in cfg.lines.items():
        src_table = tables[line_cfg.source]
        src_cols = set(src_table.schema.names)
        if line_cfg.key not in src_cols:
            col_errors.append(
                f"Line '{line_id}': key column '{line_cfg.key}' "
                f"not in source '{line_cfg.source}'"
            )
        if line_cfg.columns:
            for src_col in line_cfg.columns.values():
                if src_col not in src_cols:
                    col_errors.append(
                        f"Line '{line_id}': column '{src_col}' "
                        f"not in source '{line_cfg.source}'"
                    )
    for pid, pat_cfg in cfg.patterns.items():
        entity_src = cfg.lines.get(pat_cfg.entity_line)
        if entity_src:
            src_table = tables.get(entity_src.source)
            if src_table:
                src_cols = set(src_table.schema.names)
                if pat_cfg.event_dimensions:
                    for ed in pat_cfg.event_dimensions:
                        if ed.column not in src_cols:
                            col_errors.append(
                                f"Pattern '{pid}': event_dimension column "
                                f"'{ed.column}' not in source"
                            )
                if isinstance(pat_cfg.relations, list):
                    for rel in pat_cfg.relations:
                        if rel.key_on_entity and rel.key_on_entity not in src_cols:
                            col_errors.append(
                                f"Pattern '{pid}': relation key_on_entity "
                                f"'{rel.key_on_entity}' not in source"
                            )
        # Validate explicit anchor_fk on derived dimension groups
        if pat_cfg.derived_dimensions:
            for group in pat_cfg.derived_dimensions:
                if group.anchor_fk and group.from_pattern:
                    # Resolve event line to find source table
                    evt_line_id = None
                    from_pat = cfg.patterns.get(group.from_pattern)
                    if from_pat:
                        evt_line_id = from_pat.entity_line
                    elif group.from_pattern in cfg.lines:
                        evt_line_id = group.from_pattern
                    if evt_line_id:
                        evt_src = cfg.lines.get(evt_line_id)
                        if evt_src:
                            evt_table = tables.get(evt_src.source)
                            if evt_table:
                                evt_cols = set(evt_table.schema.names)
                                if group.anchor_fk not in evt_cols:
                                    col_errors.append(
                                        f"Pattern '{pid}': anchor_fk "
                                        f"'{group.anchor_fk}' not in "
                                        f"event source '{evt_line_id}'"
                                    )
    if col_errors:
        raise ValueError(
            "Column validation failed:\n  " + "\n  ".join(col_errors)
        )

    # 2. Register lines
    for line_id, line_cfg in cfg.lines.items():
        _add_line(builder, line_id, line_cfg, tables)

    # 3. Register patterns (dedup derived dims by anchor_line + dim_name)
    _registered_dims: set[tuple[str, str]] = set()
    for pid, pat_cfg in cfg.patterns.items():
        _add_pattern(builder, pid, pat_cfg, cfg, _registered_dims)

    # 4. Register composite lines
    for cl_id, cl_cfg in cfg.composite_lines.items():
        _add_composite_line(builder, cl_id, cl_cfg, cfg)

    # 4b. Chain lines (with pickle cache for expensive extraction)
    if not no_chains:
        chain_cache_dir = base_dir / ".cache"
        for cl_id, cl_cfg in cfg.chain_lines.items():
            t_phase = time.time()
            if verbose:
                print(f"  Extracting chains for '{cl_id}'...")
            _add_chain_line(builder, cl_id, cl_cfg, tables, verbose,
                            cache_dir=chain_cache_dir)
            if verbose:
                print(f"  Chains total: {time.time() - t_phase:.1f}s")
    elif verbose and cfg.chain_lines:
        print(f"  Skipping {len(cfg.chain_lines)} chain line(s) (--no-chains)")

    # 4c. Register aliases
    _add_aliases(builder, cfg.aliases)

    # 5. Build
    # Suppress edge table emission if --no-edges
    if no_edges:
        builder._no_edges = True

    t_phase = time.time()
    if verbose:
        print("  Building geometry...")
    builder.build()
    if verbose:
        print(f"  Geometry total: {time.time() - t_phase:.1f}s")

    # 6. Temporal
    if not no_temporal:
        t_phase = time.time()
        for tc in cfg.temporal:
            if verbose:
                print(
                    f"  Building temporal for '{tc.pattern}' "
                    f"({tc.window} windows)..."
                )
            _build_temporal(builder, tc)
        if verbose and cfg.temporal:
            print(f"  Temporal total: {time.time() - t_phase:.1f}s")

    # 6b. Persist timestamp_col from temporal configs into sphere.json
    #     Written to both the anchor pattern (named in tc.pattern) and the
    #     event pattern whose entity_line matches tc.event_line — aggregate()
    #     looks up timestamp_col on the event pattern for time_from/time_to.
    if cfg.temporal:
        sphere_path = Path(output_path) / "_gds_meta" / "sphere.json"
        sphere_data = json.loads(sphere_path.read_text())
        for tc in cfg.temporal:
            patterns = sphere_data.get("patterns", {})
            if tc.pattern in patterns:
                patterns[tc.pattern]["timestamp_col"] = tc.timestamp_col
            # Also tag the event pattern that owns tc.event_line
            for pid, pat in patterns.items():
                if pat.get("entity_line") == tc.event_line and pat.get("pattern_type") == "event":
                    pat["timestamp_col"] = tc.timestamp_col
        sphere_path.write_text(json.dumps(sphere_data, indent=2))

    # 7. Trajectory indices — now built inline during build_temporal (step 6).
    #    Standalone rebuild: GDSWriter.build_trajectory_index(pattern_id)

    if verbose:
        print(f"  Build total: {time.time() - t_total:.1f}s")


def _add_line(
    builder: object,
    line_id: str,
    line_cfg: LineConfig,
    tables: dict,
) -> None:
    """Add a line to the builder, handling column rename and fts."""
    import pyarrow as pa

    table: pa.Table = tables[line_cfg.source]

    # Column rename/select
    if line_cfg.columns:
        # columns: {target_name: source_name} — select and rename
        source_cols = list(line_cfg.columns.values())
        target_names = list(line_cfg.columns.keys())
        # Validate all source columns exist
        for src_col in source_cols:
            if src_col not in table.schema.names:
                raise ValueError(
                    f"Line '{line_id}': column '{src_col}' not found "
                    f"in source. Available: {table.schema.names}"
                )
        table = table.select(source_cols)
        table = table.rename_columns(target_names)

    # Resolve fts
    fts_columns: list[str] | str | None = None
    if isinstance(line_cfg.fts, bool):
        if line_cfg.fts:
            fts_columns = "all"
    elif isinstance(line_cfg.fts, list):
        fts_columns = line_cfg.fts

    builder.add_line(  # type: ignore[union-attr]
        line_id,
        table,
        key_col=line_cfg.key,
        role=line_cfg.role,
        fts_columns=fts_columns,
        partition_col=line_cfg.partition_col,
        description=line_cfg.description,
        source_id=line_cfg.source,
    )


def _add_aliases(
    builder: object,
    aliases: dict[str, AliasConfig],
) -> None:
    """Bridge alias configs from YAML to GDSBuilder.add_alias()."""
    for alias_id, acfg in aliases.items():
        kwargs: dict[str, Any] = {
            "base_pattern_id": acfg.base_pattern,
        }
        if acfg.cutting_plane_normal is not None:
            kwargs["cutting_plane_normal"] = acfg.cutting_plane_normal
            kwargs["cutting_plane_bias"] = acfg.cutting_plane_bias
        else:
            kwargs["cutting_plane_dimension"] = acfg.cutting_plane_dimension
            kwargs["cutting_plane_threshold"] = acfg.cutting_plane_threshold
        if acfg.description:
            kwargs["description"] = acfg.description
        builder.add_alias(alias_id, **kwargs)  # type: ignore[union-attr]


def _add_pattern(
    builder: object,
    pid: str,
    pat_cfg: PatternConfig,
    cfg: SphereConfig,
    _registered_dims: set[tuple[str, str]] | None = None,
) -> None:
    """Register a pattern with its relations, event dims, and derived dims."""
    from hypertopos.builder.builder import RelationSpec

    # Build relations list
    relations: list[RelationSpec] = []
    if isinstance(pat_cfg.relations, list):
        for rc in pat_cfg.relations:
            relations.append(RelationSpec(
                line_id=rc.line,
                fk_col=rc.key_on_entity,
                direction=rc.direction,
                required=rc.required,
                display_name=rc.display_name,
                edge_max=rc.edge_max,
            ))
    # "auto" or None -> empty relations (derived dims auto-generate)

    # Edge table config (YAML → builder)
    from hypertopos.builder.builder import EdgeTableConfig as _ETC
    _et_cfg = None
    if pat_cfg.edge_table:
        _et_cfg = _ETC(
            from_col=pat_cfg.edge_table.from_col,
            to_col=pat_cfg.edge_table.to_col,
            timestamp_col=pat_cfg.edge_table.timestamp_col,
            amount_col=pat_cfg.edge_table.amount_col,
        )

    builder.add_pattern(  # type: ignore[union-attr]
        pid,
        pattern_type=pat_cfg.type,
        entity_line=pat_cfg.entity_line,
        relations=relations,
        anomaly_percentile=pat_cfg.anomaly_percentile,
        tracked_properties=pat_cfg.tracked_properties,
        group_by_property=pat_cfg.group_by_property,
        dimension_weights=pat_cfg.dimension_weights,
        gmm_n_components=pat_cfg.gmm_n_components,
        use_mahalanobis=pat_cfg.use_mahalanobis,
        description=pat_cfg.description,
        edge_table=_et_cfg,
    )

    # Event dimensions
    if pat_cfg.event_dimensions:
        for ed in pat_cfg.event_dimensions:
            builder.add_event_dimension(  # type: ignore[union-attr]
                pid,
                column=ed.column,
                edge_max="auto",
                display_name=ed.display_name,
            )

    # Precomputed dimensions
    if pat_cfg.precomputed_dimensions:
        for pc in pat_cfg.precomputed_dimensions:
            builder.add_precomputed_dimension(  # type: ignore[union-attr]
                pat_cfg.entity_line,
                dimension_name=pc.column,
                edge_max=pc.edge_max,
                percentile=pc.percentile,
                display_name=pc.display_name,
            )

    # Graph features
    if pat_cfg.graph_features:
        gf = pat_cfg.graph_features
        builder.add_graph_features(  # type: ignore[union-attr]
            pat_cfg.entity_line,
            event_line=gf.event_line,
            from_col=gf.from_col,
            to_col=gf.to_col,
            features=gf.features,
        )

    # Derived dimensions (skip if already registered for same anchor+dim)
    if pat_cfg.derived_dimensions:
        for group in pat_cfg.derived_dimensions:
            _add_derived_group(builder, pid, group, pat_cfg, cfg, _registered_dims)


def _resolve_anchor_fk(
    anchor_line: str,
    event_line: str | None,
    from_pattern: str | None,
    cfg: SphereConfig,
) -> str:
    """Find the FK column on event_line that references anchor_line.

    Search order:
    1. If from_pattern points to a pattern with relations, use key_on_entity
       for the relation targeting anchor_line
    2. Try common column names: {anchor_line}_id, {anchor_line}, account_id
    3. Fall back to anchor_line name (best guess)
    """
    if from_pattern:
        pat = cfg.patterns.get(from_pattern)
        if pat and pat.relations and isinstance(pat.relations, list):
            for rel in pat.relations:
                if rel.line == anchor_line and rel.key_on_entity:
                    return rel.key_on_entity

    # Heuristic: naming convention for FK column
    singular = anchor_line.rstrip("s") if anchor_line.endswith("s") else anchor_line
    return f"{singular}_id"


def _add_derived_group(
    builder: object,
    pid: str,
    group: DerivedDimGroup,
    pat_cfg: PatternConfig,
    cfg: SphereConfig,
    _registered_dims: set[tuple[str, str]] | None = None,
) -> None:
    """Add all features in a derived dimension group."""
    # Resolve event_line from from_pattern (can be a pattern ID or line ID)
    event_line: str | None = None
    if group.from_pattern:
        from_pat = cfg.patterns.get(group.from_pattern)
        if from_pat:
            event_line = from_pat.entity_line
        elif group.from_pattern in cfg.lines:
            event_line = group.from_pattern

    # Resolve time_col from temporal config for this pattern
    time_col: str | None = None
    for tc in cfg.temporal:
        if tc.pattern == pid:
            time_col = tc.timestamp_col
            break

    for feat in group.features:
        dim_key = (pat_cfg.entity_line, feat.dimension_name)
        if _registered_dims is not None:
            if dim_key in _registered_dims:
                continue
            _registered_dims.add(dim_key)
        _add_one_feature(
            builder, pid, feat, pat_cfg, event_line, time_col,
            from_pattern=group.from_pattern, cfg=cfg,
            explicit_anchor_fk=group.anchor_fk,
        )


def _add_one_feature(
    builder: object,
    pid: str,
    feat: FeatureSpec,
    pat_cfg: PatternConfig,
    event_line: str | None,
    time_col: str | None,
    from_pattern: str | None = None,
    cfg: SphereConfig | None = None,
    explicit_anchor_fk: str | None = None,
) -> None:
    """Add a single derived dimension feature to the builder."""
    if not event_line:
        raise ValueError(
            f"Pattern '{pid}' derived_dimensions: cannot resolve event_line. "
            "Specify 'from_pattern' in the derived_dimensions group."
        )

    # For temporal or IET features, use the time_col from temporal config
    is_iet = feat.metric.startswith("iet_")
    feat_time_col = feat.time_col or (
        time_col if (feat.time_window or is_iet) else None
    )

    # Resolve anchor FK: use explicit override if provided, else auto-resolve
    if explicit_anchor_fk:
        anchor_fk = explicit_anchor_fk
    else:
        anchor_fk = _resolve_anchor_fk(
            pat_cfg.entity_line, event_line, from_pattern, cfg,
        )

    builder.add_derived_dimension(  # type: ignore[union-attr]
        anchor_line=pat_cfg.entity_line,
        event_line=event_line,
        anchor_fk=anchor_fk,
        metric=feat.metric,
        metric_col=feat.metric_col,
        dimension_name=feat.dimension_name,
        edge_max="auto",
        percentile=99.0,
        time_col=feat_time_col,
        time_window=feat.time_window,
        window_aggregation=feat.window_aggregation,
    )


def _chain_cache_key(cl_id: str, cl_cfg: ChainLineConfig, n_events: int) -> str:
    """Deterministic cache key from chain extraction parameters."""
    import hashlib
    parts = [
        cl_id, cl_cfg.from_col, cl_cfg.to_col,
        str(cl_cfg.seed_percentile_fan_out),
        str(cl_cfg.seed_percentile_cross_bank),
        str(cl_cfg.seed_multi_currency),
        str(cl_cfg.seed_pass_through),
        str(cl_cfg.time_window_hours),
        str(cl_cfg.max_hops), str(cl_cfg.min_hops),
        str(cl_cfg.max_chains), str(cl_cfg.bidirectional),
        str(n_events),
    ]
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"chains_{cl_id}_{h}.pkl"


def _add_chain_line(
    builder: object,
    cl_id: str,
    cl_cfg: ChainLineConfig,
    tables: dict,
    verbose: bool,
    cache_dir: Path | None = None,
) -> None:
    """Extract chains and register as anchor line."""
    import numpy as np

    from hypertopos.engine.chains import extract_chains, parse_timestamps_to_epoch

    # Find the event line source table
    event_table = None
    for _src_name, tbl in tables.items():
        if cl_cfg.from_col in tbl.schema.names and cl_cfg.to_col in tbl.schema.names:
            event_table = tbl
            break
    if event_table is None:
        raise ValueError(
            f"chain_lines.{cl_id}: cannot find source with columns "
            f"'{cl_cfg.from_col}' and '{cl_cfg.to_col}'"
        )

    import pyarrow.compute as pc

    n = event_table.num_rows

    # Check cache
    import pickle
    chains = None
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / _chain_cache_key(cl_id, cl_cfg, n)
        if cache_file.exists():
            if verbose:
                print(f"    Loading cached chains from {cache_file.name}...")
            chains = pickle.loads(cache_file.read_bytes())
            if verbose:
                print(f"    Chains loaded: {len(chains):,}")

    if chains is None:
        # Bulk extract all needed columns in one pass
        needed_cols = [cl_cfg.from_col, cl_cfg.to_col]
        if "primary_key" in event_table.schema.names:
            needed_cols.append("primary_key")

        ts_col = None
        for col_name in ("timestamp", "date", "orderdate"):
            if col_name in event_table.schema.names:
                ts_col = col_name
                needed_cols.append(ts_col)
                break
        # Type-based fallback: first column with timestamp type
        if ts_col is None:
            import pyarrow as pa
            for field in event_table.schema:
                if pa.types.is_timestamp(field.type):
                    ts_col = field.name
                    needed_cols.append(ts_col)
                    break

        cat_col = None
        for col in ("receiving_currency", "category", "k_symbol", "type"):
            if col in event_table.schema.names:
                cat_col = col
                needed_cols.append(cat_col)
                break

        amt_col = None
        for col in ("amount_received", "amount", "extendedprice", "fare_amount", "total_amount"):
            if col in event_table.schema.names:
                amt_col = col
                needed_cols.append(amt_col)
                break

        # Single to_pydict — one pass instead of 6 separate to_pylist calls
        bulk = event_table.select(list(set(needed_cols))).to_pydict()
        from_keys = bulk[cl_cfg.from_col]
        to_keys = bulk[cl_cfg.to_col]

        # Timestamps — Arrow vectorized path
        timestamps = parse_timestamps_to_epoch(bulk[ts_col]) if ts_col else [0.0] * n

        # PKs
        if "primary_key" in bulk:
            event_pks = bulk["primary_key"]
        else:
            event_pks = [f"EV-{i:08d}" for i in range(n)]

        # Categories
        categories = bulk.get(cat_col, [""] * n) if cat_col else [""] * n

        # Amounts — vectorized null handling
        if amt_col:
            amt_arr = event_table[amt_col]
            import pyarrow as pa
            amt_filled = pc.fill_null(pc.cast(amt_arr, pa.float64()), 0.0)
            amounts = amt_filled.to_pylist()
        else:
            amounts = [0.0] * n

        # Seed selection — Arrow groupby instead of Python loop
        # Fan-out: count distinct targets per source
        fan_table = event_table.select(
            [cl_cfg.from_col, cl_cfg.to_col],
        ).group_by(cl_cfg.from_col).aggregate(
            [(cl_cfg.to_col, "count_distinct")]
        )
        fan_col = f"{cl_cfg.to_col}_count_distinct"
        fan_keys = fan_table[cl_cfg.from_col].to_pylist()
        fan_vals = fan_table[fan_col].to_pylist()
        fan_map = dict(zip(fan_keys, fan_vals, strict=False))

        t_fan = float(np.percentile(fan_vals or [0], cl_cfg.seed_percentile_fan_out))

        # Currency diversity per source (if cat_col exists)
        curr_map: dict[str, int] = {}
        if cat_col:
            curr_table = event_table.select(
                [cl_cfg.from_col, cat_col],
            ).group_by(cl_cfg.from_col).aggregate(
                [(cat_col, "count_distinct")]
            )
            curr_keys = curr_table[cl_cfg.from_col].to_pylist()
            curr_vals = curr_table[f"{cat_col}_count_distinct"].to_pylist()
            curr_map = dict(zip(curr_keys, curr_vals, strict=False))

        # In-sources count (distinct senders per receiver)
        in_table = event_table.select(
            [cl_cfg.to_col, cl_cfg.from_col],
        ).group_by(cl_cfg.to_col).aggregate(
            [(cl_cfg.from_col, "count_distinct")]
        )
        in_keys = in_table[cl_cfg.to_col].to_pylist()
        in_vals = in_table[f"{cl_cfg.from_col}_count_distinct"].to_pylist()
        in_map = dict(zip(in_keys, in_vals, strict=False))

        # Out count (total events per source)
        out_table = event_table.group_by(cl_cfg.from_col).aggregate(
            [(cl_cfg.to_col, "count")]
        )
        out_keys = out_table[cl_cfg.from_col].to_pylist()
        out_vals = out_table[f"{cl_cfg.to_col}_count"].to_pylist()
        out_count_map = dict(zip(out_keys, out_vals, strict=False))

        # Build seed set
        all_accts = sorted(set(from_keys) | set(to_keys) - {None})
        seed_set: set[str] = set()
        for k in all_accts:
            if (
                fan_map.get(k, 0) >= t_fan
                or curr_map.get(k, 0) >= cl_cfg.seed_multi_currency
                or (
                    cl_cfg.seed_pass_through
                    and in_map.get(k, 0) >= 2
                    and fan_map.get(k, 0) >= 2
                    and out_count_map.get(k, 0) >= 5
                )
            ):
                seed_set.add(k)

        seed_keys = sorted(seed_set)
        if verbose:
            print(f"    Seeds: {len(seed_keys):,}")

        chains = extract_chains(
            from_keys=from_keys,
            to_keys=to_keys,
            event_pks=event_pks,
            timestamps=timestamps,
            categories=categories,
            amounts=amounts,
            time_window_hours=cl_cfg.time_window_hours,
            max_hops=cl_cfg.max_hops,
            min_hops=cl_cfg.min_hops,
            seed_nodes=seed_keys,
            max_chains=cl_cfg.max_chains,
            bidirectional=cl_cfg.bidirectional,
        )

        # Save to cache
        if cache_file is not None:
            cache_file.write_bytes(pickle.dumps(chains))
            if verbose:
                print(f"    Chains cached to {cache_file.name}")

    if verbose:
        print(f"    Chains extracted: {len(chains):,}")

    chain_dicts = [c.to_dict() for c in chains]
    builder.add_chain_line(  # type: ignore[union-attr]
        cl_id,
        chains=chain_dicts,
        features=cl_cfg.features,
    )
    # Auto-create pattern for chain line
    builder.add_pattern(  # type: ignore[union-attr]
        f"{cl_id}_pattern",
        pattern_type="anchor",
        entity_line=cl_id,
        relations=[],
        anomaly_percentile=cl_cfg.anomaly_percentile,
        description=cl_cfg.description,
    )


def _add_composite_line(
    builder: object,
    cl_id: str,
    cl_cfg: CompositeLineConfig,
    cfg: SphereConfig,
) -> None:
    """Register a composite line and its derived dimensions + pattern."""
    builder.add_composite_line(  # type: ignore[union-attr]
        cl_id,
        event_line=cl_cfg.event_line,
        key_cols=cl_cfg.key_cols,
        separator=cl_cfg.separator,
    )

    # Auto-create an anchor pattern for the composite line
    composite_pid = f"{cl_id}_pattern"

    builder.add_pattern(  # type: ignore[union-attr]
        composite_pid,
        pattern_type="anchor",
        entity_line=cl_id,
        relations=[],
        anomaly_percentile=cl_cfg.anomaly_percentile,
        dimension_weights=cl_cfg.dimension_weights,
        description=cl_cfg.description,
    )

    # Add derived dimensions for composite line
    if cl_cfg.derived_dimensions:
        for group in cl_cfg.derived_dimensions:
            event_line = cl_cfg.event_line
            for feat in group.features:
                builder.add_derived_dimension(  # type: ignore[union-attr]
                    anchor_line=cl_id,
                    event_line=event_line,
                    anchor_fk=cl_cfg.key_cols,
                    metric=feat.metric,
                    metric_col=feat.metric_col,
                    dimension_name=feat.dimension_name,
                    edge_max="auto",
                    percentile=99.0,
                )


def _build_temporal(builder: object, tc: TemporalConfig) -> None:
    """Call build_temporal on the builder for one temporal config."""
    builder.build_temporal(  # type: ignore[union-attr]
        time_col=tc.timestamp_col,
        time_window=tc.window,
        event_line=tc.event_line,
        anchor_pattern=tc.pattern,
    )


def _build_trajectory_indices(
    cfg: SphereConfig,
    output_path: str,
    verbose: bool,
) -> None:
    """Build trajectory index for each anchor pattern with temporal config."""
    if not cfg.temporal:
        return

    from hypertopos.storage.writer import GDSWriter

    writer = GDSWriter(output_path)
    temporal_patterns = {tc.pattern for tc in cfg.temporal}

    for pid in temporal_patterns:
        pat = cfg.patterns.get(pid)
        if pat and pat.type == "anchor":
            if verbose:
                print(f"  Building trajectory index for '{pid}'...")
            writer.build_trajectory_index(pid)
