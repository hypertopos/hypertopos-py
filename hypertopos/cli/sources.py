# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Source loader for ``hypertopos build``.

Three tiers of data ingestion:
  1.  Single file (parquet, CSV, Arrow IPC) with optional transforms
  1b. CSV with custom delimiter / encoding
  2.  Multi-file join (PyArrow ``table.join``)
  3.  Python script with ``prepare() -> pa.Table``
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

from hypertopos.cli.schema import SourceConfig, TransformSpec

# ── type cast mapping ────────────────────────────────────────────────

_CAST_MAP: dict[str, pa.DataType] = {
    "string": pa.string(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
}


def load_source(config: SourceConfig, base_dir: Path) -> pa.Table:
    """Load a data source from config. Returns a PyArrow Table.

    Args:
        config: Parsed source configuration.
        base_dir: Directory to resolve relative paths against
                  (typically the YAML file's parent directory).
    """
    if config.script:
        return _load_script(config.script, base_dir)

    if not config.path:
        raise ValueError("Source must specify 'path' or 'script'")

    # Tier 1 / 1b: load base file
    table = _load_file(config.path, base_dir, config)

    # Tier 2: apply joins
    for join_spec in config.join:
        other = _load_file(join_spec.file, base_dir)
        if join_spec.columns:
            # Keep only join key + requested columns
            keep = list({join_spec.on} | set(join_spec.columns))
            keep = [c for c in keep if c in other.schema.names]
            other = other.select(keep)
        jt = {"left": "left outer", "inner": "inner"}.get(join_spec.type, join_spec.type)
        table = table.join(other, keys=join_spec.on, join_type=jt)

    # Tier 1c: apply column transforms (after joins)
    if config.transform:
        table = _apply_transforms(table, config.transform)

    return table


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    return p


def _load_file(
    path_str: str,
    base_dir: Path,
    config: SourceConfig | None = None,
) -> pa.Table:
    """Load a single data file."""
    p = _resolve_path(path_str, base_dir)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")

    suffix = "".join(p.suffixes).lower()
    fmt = (config.format if config else None) or ""

    if fmt == "csv" or suffix in (".csv", ".csv.gz", ".tsv"):
        return _load_csv(p, config)
    if suffix in (".parquet", ".pq"):
        return pq.ParquetFile(str(p)).read()
    if suffix in (".arrow", ".ipc", ".feather"):
        reader = pa_ipc.open_file(str(p))
        return reader.read_all()

    # Try parquet as default for unknown extensions
    try:
        return pq.ParquetFile(str(p)).read()
    except Exception as exc:
        raise ValueError(
            f"Cannot determine format for '{p}'. "
            "Supported: .parquet, .pq, .csv, .csv.gz, .tsv, .arrow, .ipc, .feather"
        ) from exc


def _load_csv(p: Path, config: SourceConfig | None) -> pa.Table:
    """Load CSV with optional delimiter and encoding."""
    parse_options = pa_csv.ParseOptions()
    read_options = pa_csv.ReadOptions()

    if config and config.delimiter:
        parse_options = pa_csv.ParseOptions(delimiter=config.delimiter)
    if config and config.encoding:
        read_options = pa_csv.ReadOptions(encoding=config.encoding)

    return pa_csv.read_csv(str(p), parse_options=parse_options,
                           read_options=read_options)


def _apply_transforms(
    table: pa.Table,
    transforms: dict[str, TransformSpec],
) -> pa.Table:
    """Apply type casts and fill_null to table columns."""
    for col_name, tspec in transforms.items():
        if col_name not in table.schema.names:
            continue

        col = table.column(col_name)

        # fill_null first (before cast, so fill value type matches source)
        if tspec.fill_null is not None:
            fill_val = tspec.fill_null
            # Cast fill_val scalar to match column type for if_else
            col = _fill_null_column(col, fill_val)

        # type cast
        if tspec.type:
            target = _CAST_MAP.get(tspec.type)
            if target is None:
                raise ValueError(
                    f"Unknown cast type '{tspec.type}' for column '{col_name}'"
                )
            col = pc.cast(col, target, safe=False)

        idx = table.schema.get_field_index(col_name)
        table = table.set_column(idx, col_name, col)

    return table


def _fill_null_column(col: pa.ChunkedArray, fill_val: Any) -> pa.ChunkedArray:
    """Replace nulls in a column with a fill value."""
    try:
        return pc.if_else(pc.is_null(col), fill_val, col)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        # If types don't match, try casting fill_val
        scalar = pa.scalar(fill_val, type=col.type)
        return pc.if_else(pc.is_null(col), scalar, col)


def _load_script(script_path: str, base_dir: Path) -> pa.Table:
    """Load data via a Python script's ``prepare()`` function.

    Results are cached as parquet in ``.cache/`` next to the script.
    Cache is invalidated when the script's mtime changes.
    """
    p = _resolve_path(script_path, base_dir)
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")

    # Check cache (parquet + mtime marker)
    cache_dir = p.parent / ".cache"
    cache_parquet = cache_dir / f"{p.stem}.parquet"
    cache_mtime = cache_dir / f"{p.stem}.mtime"
    script_mtime = str(p.stat().st_mtime)

    if (
        cache_parquet.exists()
        and cache_mtime.exists()
        and cache_mtime.read_text().strip() == script_mtime
    ):
        return pq.ParquetFile(str(cache_parquet)).read()

    # Cache miss — run script
    module_name = p.stem
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load script '{p}' as a Python module")

    # Add script's directory to sys.path so it can import local modules
    script_dir = str(p.parent)
    path_added = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        path_added = True

    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        if not hasattr(module, "prepare"):
            raise ValueError(
                f"Script '{p}' must define a 'prepare()' function "
                "that returns a pyarrow.Table"
            )

        result = module.prepare()
        if not isinstance(result, pa.Table):
            raise ValueError(
                f"Script '{p}' prepare() must return a pyarrow.Table, "
                f"got {type(result).__name__}"
            )

        # Write cache
        cache_dir.mkdir(exist_ok=True)
        pq.write_table(result, str(cache_parquet))
        cache_mtime.write_text(script_mtime)

        return result
    finally:
        if path_added:
            sys.path.remove(script_dir)
