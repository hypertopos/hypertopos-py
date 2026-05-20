# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Read-side support for the conformance sidecar (M1.7).

The builder writes per-pattern violations as a Lance dataset under
``_gds_meta/conformance/violations/{pattern_id}/v={N}.lance`` next to a
``manifest.json`` carrying the rule-set hash. This module reads them with
filter pushdown for the navigator's ``find_conformance_violations``
primitive.

Bail-guard: the sidecar directory's existence is a cheap O(1) filesystem
check; we never open the Lance dataset when the sidecar is absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lance as _lance  # type: ignore[import-untyped]

# Severity ordering — keep in sync with ``ConformanceRule.severity`` Literal.
_SEVERITY_RANK: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def _sidecar_dir(base_path: Path, pattern_id: str) -> Path:
    return (
        base_path
        / "_gds_meta"
        / "conformance"
        / "violations"
        / pattern_id
    )


def _dataset_path(base_path: Path, pattern_id: str, version: int) -> Path:
    return _sidecar_dir(base_path, pattern_id) / f"v={version}.lance"


def _read_manifest(base_path: Path, pattern_id: str) -> dict[str, Any] | None:
    manifest_path = _sidecar_dir(base_path, pattern_id) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def read_violations(
    base_path: Path,
    pattern_id: str,
    version: int,
    rule_id: str | None = None,
    severity_min: str = "low",
    top_n: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Read the sidecar Lance dataset for ``pattern_id`` at ``version``.

    Filters:
      * ``rule_id`` — exact-match Lance pushdown when given;
      * ``severity_min`` — keep rows with severity rank >= the floor.
        Filtered post-scan because severity is a categorical string and
        the rank order does not align with Lance's lexicographic
        string ordering.

    Returns ``(violations, manifest)``. Each violation is a dict with
    ``primary_key``, ``rule_id``, ``severity``. ``manifest`` is ``None``
    when the sidecar directory does not exist (cheap-metadata bail-guard).
    """
    if severity_min not in _SEVERITY_RANK:
        raise ValueError(
            f"severity_min must be one of {list(_SEVERITY_RANK)!r}; "
            f"got {severity_min!r}",
        )

    # ── Bail-guard: cheap metadata check first. ─────────────────────────
    side_dir = _sidecar_dir(base_path, pattern_id)
    if not side_dir.exists():
        return ([], None)

    manifest = _read_manifest(base_path, pattern_id)

    ds_path = _dataset_path(base_path, pattern_id, version)
    if not ds_path.exists():
        # Sidecar exists for the pattern but not for this version — e.g.
        # the version was rebuilt without rules. Return empty + the
        # manifest so the caller can still surface the hash for
        # observability.
        return ([], manifest)

    ds = _lance.dataset(str(ds_path))
    columns = ["primary_key", "rule_id", "severity"]

    # Lance ``to_table`` supports ``filter`` pushdown via a SQL fragment
    # string. We push ``rule_id`` (exact eq) when given; severity is
    # filtered post-scan against the rank order.
    if rule_id is not None:
        # Single-quote the literal; rule_id is constrained by upstream
        # validation, but we still escape embedded quotes defensively.
        escaped = rule_id.replace("'", "''")
        table = ds.to_table(
            columns=columns,
            filter=f"rule_id = '{escaped}'",
        )
    else:
        table = ds.to_table(columns=columns)

    floor = _SEVERITY_RANK[severity_min]
    pks = table.column("primary_key").to_pylist()
    rids = table.column("rule_id").to_pylist()
    sevs = table.column("severity").to_pylist()

    violations: list[dict[str, Any]] = []
    for pk, rid, sev in zip(pks, rids, sevs, strict=False):
        sev_rank = _SEVERITY_RANK.get(sev, -1)
        if sev_rank < floor:
            continue
        violations.append(
            {"primary_key": pk, "rule_id": rid, "severity": sev},
        )
        if len(violations) >= top_n:
            break

    return (violations, manifest)
