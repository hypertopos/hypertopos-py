# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Cloud-ops verbs for ``hypertopos sphere`` — health, validate, diff.

These commands operate on a *built* sphere directory (the one ``info``
reads), not on a pre-build ``sphere.yaml`` config. They are shaped for
CI gates and pre-deploy checks:

- ``health`` composes ``Navigator.sphere_overview`` + ``Navigator.check_alerts``
  and exits 2 on critical (HIGH-severity) alerts when asked.
- ``validate`` runs structural integrity checks over the built sphere;
  ``--strict`` promotes calibration / dimension-quality warnings to errors.
- ``diff`` reports the pattern-inventory delta and per-pattern calibration
  drift between two sphere directories — a pre-deploy regression gate.
- ``ingest`` appends a new-/changed-entities table to one pattern's geometry
  via ``GDSBuilder.incremental_update`` (optionally finalizing the batched
  rank recompute + ANN rebuild) so ops can run incremental ingest without
  writing Python.

Every command supports ``--json`` for machine-readable output. In JSON
mode, stdout carries ONLY the JSON document; human-readable diagnostics
and errors go to stderr.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hypertopos._file_formats import normalized_suffix


def _sanitize_for_json(obj: Any) -> Any:
    """Replace non-finite floats (``±inf`` / ``NaN``) with ``None`` recursively.

    Mirrors the MCP ``observability._sanitize_for_json`` helper so the CLI
    and MCP surfaces emit identical strict-JSON-safe payloads. ``json.dumps``
    emits bare ``Infinity`` / ``NaN`` literals for non-finite floats, which
    strict parsers (``jq``, most JSON libraries) reject. Navigator math can
    produce ``NaN`` on degenerate inputs — e.g. ``overall_drift_rms`` over a
    zero-dimension pattern, or a ``theta_norm`` over a NaN-tainted theta.
    Also catches ``np.floating`` so a future field that skips the ``float()``
    cast cannot leak a ``"nan"`` string through ``default=str``.
    """
    if isinstance(obj, (float, np.floating)) and not math.isfinite(float(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _dump_json(payload: Any) -> str:
    """Serialize a payload to indented, strict-JSON-safe text."""
    return json.dumps(_sanitize_for_json(payload), indent=2, default=str)


def _sphere_json_path(sphere_path: str) -> Path:
    """Return the sphere.json path for a sphere directory, or exit 1."""
    p = Path(sphere_path)
    meta = p / "_gds_meta" / "sphere.json"
    if not meta.exists():
        print(
            f"error: not a sphere directory (no _gds_meta/sphere.json): {p}",
            file=sys.stderr,
        )
        sys.exit(1)
    return meta


def _open_navigator(sphere_path: str) -> Any:
    """Open a built sphere and return a fresh navigator.

    Validates the directory first (exits 1 on a non-sphere path) so the
    error message matches ``hypertopos info``.
    """
    _sphere_json_path(sphere_path)
    from hypertopos.sphere import HyperSphere

    sphere = HyperSphere.open(sphere_path)
    session = sphere.session("cli")
    return session.navigator()


# ── sphere health ────────────────────────────────────────────────────


def run_sphere_health(
    sphere_path: str,
    as_json: bool,
    exit_code_on_critical: bool,
) -> None:
    """Health-check verb for CI gates.

    Composes ``sphere_overview`` (population summary per pattern) and
    ``check_alerts`` (6 geometric health checks). ``status`` is derived
    from alert severity:

    - ``"critical"`` — at least one HIGH-severity alert.
    - ``"warning"`` — at least one MEDIUM-severity alert (no HIGH).
    - ``"ok"`` — no alerts.

    With ``--exit-code-on-critical`` the process exits 2 when status is
    ``"critical"`` (and 0 otherwise), so ``set -e`` shell gates can fail
    a deploy on a critical sphere. Without the flag the exit code is
    always 0 (a non-sphere path still exits 1).
    """
    nav = _open_navigator(sphere_path)
    overview = nav.sphere_overview()
    alerts_result = nav.check_alerts()
    alerts = alerts_result.get("alerts", [])

    severities = {a.get("severity") for a in alerts}
    if "HIGH" in severities:
        status = "critical"
    elif "MEDIUM" in severities:
        status = "warning"
    else:
        status = "ok"

    payload = {
        "status": status,
        "sphere_path": str(Path(sphere_path).resolve()),
        "overview": overview,
        "alerts": alerts_result,
    }

    if as_json:
        print(_dump_json(payload))
    else:
        print(f"Status: {status}")
        print(f"Patterns: {len(overview)}")
        n_high = sum(1 for a in alerts if a.get("severity") == "HIGH")
        n_med = sum(1 for a in alerts if a.get("severity") == "MEDIUM")
        print(f"Alerts: {len(alerts)} ({n_high} HIGH, {n_med} MEDIUM)")
        for a in alerts:
            print(
                f"  [{a.get('severity')}] {a.get('pattern_id')}: "
                f"{a.get('message')}"
            )

    if exit_code_on_critical and status == "critical":
        sys.exit(2)


# ── sphere validate ──────────────────────────────────────────────────


def run_sphere_validate(
    sphere_path: str,
    strict: bool,
    as_json: bool,
) -> None:
    """Structural integrity check over a built sphere directory.

    Always checks (errors — these always fail validation):

    - ``sphere.json`` parses as JSON.
    - each declared line has a ``points/<line_id>`` directory on disk.
    - each declared pattern has a ``geometry/<pattern_id>`` directory.

    ``--strict`` additionally promotes the soft signals already surfaced
    by ``sphere_overview`` to errors:

    - any pattern whose ``calibration_health`` is ``"suspect"`` or
      ``"poor"`` (``"good"`` is the healthy state).
    - any pattern carrying ``dim_quality_warnings``.

    Exits 0 when valid (no errors), 1 when invalid. ``--json`` emits a
    ``{valid, errors, warnings, strict}`` document on stdout.
    """
    base = Path(sphere_path)
    _sphere_json_path(sphere_path)

    errors: list[str] = []
    warnings: list[str] = []

    sphere_data = json.loads(
        (base / "_gds_meta" / "sphere.json").read_text(encoding="utf-8")
    )

    # Structural: line points + pattern geometry directories on disk.
    # Derived-dimension lines (``_d_`` prefix) are virtual — they live as
    # extra columns on their anchor's geometry, never as a materialized
    # points table — so they have no points/ directory by design.
    for line_id in sphere_data.get("lines", {}):
        if line_id.startswith("_d_"):
            continue
        if not (base / "points" / line_id).exists():
            errors.append(f"Line '{line_id}': missing points/ directory")

    for pid in sphere_data.get("patterns", {}):
        if not (base / "geometry" / pid).exists():
            errors.append(f"Pattern '{pid}': missing geometry/ directory")

    # Soft signals from the navigator overview. Collected as warnings
    # always; --strict promotes them to errors. Guarded so a sphere with
    # no patterns yields nothing here (overview == []).
    #
    # The whole soft-signal pass is wrapped: on a corrupt sphere (e.g. a
    # declared pattern whose geometry data is missing) the overview can raise,
    # and a crash here would suppress the {valid, errors, warnings} report the
    # structural checks above already populated. Convert the failure to a
    # warning so the JSON document is always emitted.
    try:
        nav = _open_navigator(sphere_path)
        overview = nav.sphere_overview()
        for entry in overview:
            pid = entry["pattern_id"]
            cal = entry.get("calibration_health")
            if cal in ("suspect", "poor"):
                warnings.append(
                    f"Pattern '{pid}': calibration_health is '{cal}'"
                )
            for w in entry.get("dim_quality_warnings", []) or []:
                warnings.append(
                    f"Pattern '{pid}': dim_quality warning "
                    f"({w.get('type', 'unknown')}) on dim "
                    f"'{w.get('dim_label', w.get('dim_index', '?'))}'"
                )
    except Exception as exc:  # noqa: BLE001 — surface as a warning, never crash
        warnings.append(
            f"Soft-signal overview unavailable: {exc}"
        )

    if strict:
        errors.extend(warnings)
        warnings = []

    valid = not errors

    if as_json:
        print(
            _dump_json(
                {
                    "valid": valid,
                    "strict": strict,
                    "errors": errors,
                    "warnings": warnings,
                }
            )
        )
    else:
        if valid:
            print(f"Valid: {sphere_data.get('sphere_id', '?')}")
            for w in warnings:
                print(f"  warning: {w}", file=sys.stderr)
        else:
            print("Validation errors:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)

    if not valid:
        sys.exit(1)


# ── sphere diff ──────────────────────────────────────────────────────


def run_sphere_diff(
    old_path: str,
    new_path: str,
    as_json: bool,
) -> None:
    """Pre-deploy diff between two built sphere directories.

    Reports two deltas:

    - **pattern inventory** — pattern ids ``added`` (in new, not old),
      ``removed`` (in old, not new), and ``common`` (in both).
    - **calibration drift** — for each common pattern, the per-dimension
      μ/σ/θ drift between the latest calibration epoch of each sphere
      (``overall_drift_rms`` + top-drifted dims). Patterns whose
      calibration schema differs between the two spheres are marked
      ``not_comparable`` rather than crashing.

    Two identical sphere paths yield ``identical: true`` with empty
    deltas. Exits 0 always (a non-sphere path still exits 1).
    """
    old_base = Path(old_path)
    new_base = Path(new_path)
    old_meta = _sphere_json_path(old_path)
    new_meta = _sphere_json_path(new_path)

    identical = old_base.resolve() == new_base.resolve()

    old_data = json.loads(old_meta.read_text(encoding="utf-8"))
    new_data = json.loads(new_meta.read_text(encoding="utf-8"))

    old_patterns = set(old_data.get("patterns", {}))
    new_patterns = set(new_data.get("patterns", {}))
    added = sorted(new_patterns - old_patterns)
    removed = sorted(old_patterns - new_patterns)
    common = sorted(old_patterns & new_patterns)

    calibration_drift: list[dict[str, Any]] = []
    if common:
        from dataclasses import asdict

        from hypertopos.navigation.navigator import _compute_calibration_drift
        from hypertopos.storage.reader import GDSReader

        old_reader = GDSReader(old_path)
        new_reader = GDSReader(new_path)
        for pid in common:
            try:
                fit_from = old_reader.read_calibration_fit(pid)
                fit_to = new_reader.read_calibration_fit(pid)
            except Exception as exc:  # noqa: BLE001 — surface as a row, never crash
                calibration_drift.append(
                    {"pattern_id": pid, "not_comparable": True, "reason": str(exc)}
                )
                continue
            if fit_from.schema_hash != fit_to.schema_hash:
                calibration_drift.append(
                    {
                        "pattern_id": pid,
                        "not_comparable": True,
                        "reason": "schema_hash mismatch — dimensions not comparable",
                    }
                )
                continue
            report = _compute_calibration_drift(
                fit_from, fit_to, top_n=10, verbose=False,
            )
            row = asdict(report)
            row["per_dimension"] = None  # always omit the full breakdown
            calibration_drift.append(row)

    payload = {
        "identical": identical,
        "old_sphere": old_data.get("sphere_id", "?"),
        "new_sphere": new_data.get("sphere_id", "?"),
        "pattern_inventory": {
            "added": added,
            "removed": removed,
            "common": common,
        },
        "calibration_drift": calibration_drift,
    }

    if as_json:
        print(_dump_json(payload))
    else:
        print(f"Old: {payload['old_sphere']}   New: {payload['new_sphere']}")
        if identical:
            print("Identical sphere paths.")
        print(f"Patterns added:   {added or '(none)'}")
        print(f"Patterns removed: {removed or '(none)'}")
        print(f"Patterns common:  {common or '(none)'}")
        for row in calibration_drift:
            pid = row["pattern_id"]
            if row.get("not_comparable"):
                print(f"  {pid}: not comparable ({row.get('reason')})")
            else:
                print(
                    f"  {pid}: overall_drift_rms="
                    f"{row.get('overall_drift_rms'):.4f}"
                )


# ── sphere ingest ────────────────────────────────────────────────────


def _load_points_table(points_path: str) -> Any:
    """Load a new-/changed-entities table, dispatching on file suffix.

    Supports the three already-supported tabular formats: Arrow IPC
    (``.arrow`` / ``.arrows``), Parquet (``.parquet`` / ``.pq``), and CSV
    (``.csv`` / ``.csv.gz``). The table must carry a ``primary_key`` column —
    ``incremental_update`` keys every changed entity by it. Exits 1 with a
    clear message on a missing file or unsupported suffix.
    """
    p = Path(points_path)
    if not p.exists():
        print(f"error: points file not found: {p}", file=sys.stderr)
        sys.exit(1)

    suffix = normalized_suffix(p)
    if suffix in (".arrow", ".arrows"):
        import pyarrow.feather as feather

        return feather.read_table(str(p))
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(p)).read()
    if suffix in (".csv", ".csv.gz"):
        import pyarrow.csv as pa_csv

        return pa_csv.read_csv(str(p))

    print(
        f"error: unsupported points format '{suffix}' for file '{points_path}'. "
        "Supported: .arrow, .arrows, .parquet, .pq, .csv, .csv.gz",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_ingest_pattern(sphere_meta: Path, pattern: str | None) -> str:
    """Resolve which pattern to ingest into, or exit 1 with a clear message.

    ``incremental_update`` is per-pattern. When ``--pattern`` is given it must
    name a declared pattern. When omitted, the sole pattern is used; a sphere
    with zero or multiple patterns requires an explicit ``--pattern``.
    """
    sphere_data = json.loads(sphere_meta.read_text(encoding="utf-8"))
    patterns = list(sphere_data.get("patterns", {}))

    if pattern is not None:
        if pattern not in patterns:
            print(
                f"error: pattern '{pattern}' not found in sphere "
                f"(declared: {patterns or '(none)'})",
                file=sys.stderr,
            )
            sys.exit(1)
        return pattern

    if len(patterns) == 1:
        return patterns[0]
    if not patterns:
        print(
            "error: sphere has no patterns to ingest into",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"error: sphere has {len(patterns)} patterns; pass --pattern to choose "
        f"one of: {patterns}",
        file=sys.stderr,
    )
    sys.exit(1)


def _delta_index_rows(lance_path: str) -> int | None:
    """Return ``num_rows_indexed`` for the delta vector index, or None.

    None means no delta vector index is present. Used to detect whether an
    incremental update rebuilt / extended the ANN index over the geometry.
    """
    import lance as _lance

    try:
        ds = _lance.dataset(lance_path)
        for idx in ds.describe_indices():
            if "delta" in idx.field_names:
                return idx.num_rows_indexed
    except Exception:  # noqa: BLE001 — absence of an index is a valid state
        return None
    return None


def run_sphere_ingest(
    sphere_path: str,
    points_path: str,
    pattern: str | None,
    recalibrate: str,
    reindex: bool,
    finalize: bool,
    as_json: bool,
) -> None:
    """Incremental-ingest verb: append a new-/changed-entities table.

    Loads the points table (Arrow IPC / Parquet / CSV), resolves the target
    pattern, then calls ``GDSBuilder.incremental_update`` to append the rows to
    that pattern's geometry. With ``--finalize`` it additionally calls
    ``finalize_incremental`` to recompute the global ``delta_rank_pct`` and
    rebuild the ANN index once at the end of a batch.

    Emits a ``{pattern_id, added, modified, deleted, population_size,
    geometry_version_before, geometry_version_after, reindexed, finalized,
    drift_pct}`` summary. ``geometry_version_*`` are the geometry Lance
    dataset versions before/after; ``reindexed`` reports whether the delta
    ANN index's coverage changed across the update. Exits 1 on a non-sphere
    path, a missing/unsupported points file, an unknown pattern, or a
    geometry-reconstruction error surfaced by the builder.
    """
    meta = _sphere_json_path(sphere_path)
    pattern_id = _resolve_ingest_pattern(meta, pattern)
    points = _load_points_table(points_path)

    lance_path = str(
        Path(sphere_path) / "geometry" / pattern_id / "data.lance"
    )
    version_before = _geometry_version(lance_path)
    index_rows_before = _delta_index_rows(lance_path)

    from hypertopos.builder.builder import GDSBuilder

    sphere_data = json.loads(meta.read_text(encoding="utf-8"))
    builder = GDSBuilder(sphere_data.get("sphere_id", "sphere"), sphere_path)
    try:
        result = builder.incremental_update(
            pattern_id,
            changed_entities=points,
            recalibrate=recalibrate,
            reindex=reindex,
        )
        if finalize:
            builder.finalize_incremental(pattern_id)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    version_after = _geometry_version(lance_path)
    index_rows_after = _delta_index_rows(lance_path)
    reindexed = index_rows_after != index_rows_before

    payload = {
        "sphere_path": str(Path(sphere_path).resolve()),
        "pattern_id": pattern_id,
        "added": result.added,
        "modified": result.modified,
        "deleted": result.deleted,
        "population_size": result.population_size,
        "drift_pct": result.drift_pct,
        "geometry_version_before": version_before,
        "geometry_version_after": version_after,
        "reindexed": reindexed,
        "finalized": finalize,
    }

    if as_json:
        print(_dump_json(payload))
    else:
        print(f"Pattern: {pattern_id}")
        print(
            f"Added: {result.added}   Modified: {result.modified}   "
            f"Deleted: {result.deleted}"
        )
        print(f"Population: {result.population_size}")
        print(
            f"Geometry version: {version_before} -> {version_after}   "
            f"reindexed={reindexed}   finalized={finalize}"
        )


def _geometry_version(lance_path: str) -> int | None:
    """Return the current geometry Lance dataset version, or None if absent."""
    import lance as _lance

    try:
        return int(_lance.dataset(lance_path).version)
    except Exception:  # noqa: BLE001 — a pattern with no geometry yet → None
        return None
