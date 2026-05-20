# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Conformance-rule evaluation for the builder (M1.7).

Compiles ``ConformancePredicate`` ASTs to PyArrow ``compute.Expression``
trees and evaluates them against a pattern's points table. Violation rows
are persisted to a sidecar Lance dataset under
``_gds_meta/conformance/violations/{pattern_id}/v={N}.lance`` together with
a ``manifest.json`` carrying ``rule_set_hash`` for change detection.

No ``eval()`` — the predicate language is intentionally minimal and the
compiler walks the AST recursively.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lance as _lance  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.compute as pc

from hypertopos.model.sphere import (
    ConformancePredicate,
    Pattern,
)

# Comparison ops → PyArrow compute functions on (field, value) pairs.
_BINARY_COMPARISONS: dict[str, str] = {
    "==": "equal",
    "!=": "not_equal",
    "<": "less",
    "<=": "less_equal",
    ">": "greater",
    ">=": "greater_equal",
}


def compile_predicate(pred: ConformancePredicate) -> pc.Expression:
    """Recursively compile a predicate AST to a PyArrow expression.

    Logical compounds (``and``/``or``/``not``) wrap child expressions via
    ``Expression.__and__`` / ``__or__`` / ``~``. Comparison leaves resolve
    ``prop`` via ``pc.field`` and apply the comparison function with the
    literal RHS. The ``in`` op uses ``Expression.isin``.

    Raises ``ValueError`` for malformed AST nodes (missing terms on a
    logical op, missing ``prop`` on a comparison, wrong arity on ``not``,
    non-iterable RHS on ``in``).
    """
    op = pred.op
    if op == "and":
        if not pred.terms:
            raise ValueError("'and' predicate requires at least one term")
        exprs = [compile_predicate(t) for t in pred.terms]
        out = exprs[0]
        for e in exprs[1:]:
            out = out & e
        return out
    if op == "or":
        if not pred.terms:
            raise ValueError("'or' predicate requires at least one term")
        exprs = [compile_predicate(t) for t in pred.terms]
        out = exprs[0]
        for e in exprs[1:]:
            out = out | e
        return out
    if op == "not":
        if not pred.terms or len(pred.terms) != 1:
            raise ValueError("'not' predicate requires exactly one term")
        return ~compile_predicate(pred.terms[0])
    if op in _BINARY_COMPARISONS:
        if pred.prop is None:
            raise ValueError(
                f"comparison op {op!r} requires a 'prop' column name",
            )
        fn_name = _BINARY_COMPARISONS[op]
        fn = getattr(pc, fn_name)
        return fn(pc.field(pred.prop), pred.value)
    if op == "in":
        if pred.prop is None:
            raise ValueError("'in' predicate requires a 'prop' column name")
        if pred.value is None or not hasattr(pred.value, "__iter__"):
            raise ValueError("'in' predicate requires an iterable 'value'")
        return pc.field(pred.prop).isin(list(pred.value))
    raise ValueError(f"unsupported predicate op {op!r}")


# Schema for the sidecar violations table. ``primary_key`` is large_string
# to match the points-table primary-key encoding used elsewhere in the
# project; cast happens at write time.
_VIOLATIONS_SCHEMA = pa.schema(
    [
        pa.field("primary_key", pa.large_string()),
        pa.field("rule_id", pa.large_string()),
        pa.field("severity", pa.large_string()),
    ]
)


def evaluate_conformance_rules(
    pattern: Pattern,
    points_table: pa.Table,
) -> pa.Table:
    """Evaluate every rule on ``pattern.conformance_rules`` against the
    ``points_table`` and return a single concatenated violations table.

    For each rule, compiles the predicate to a PyArrow expression,
    filters the points table to violating rows, and emits one row per
    violator with the rule's ``rule_id`` and ``severity``. The returned
    table is empty (zero rows, correct schema) when the pattern has no
    rules or no violations.
    """
    if not pattern.conformance_rules:
        return _VIOLATIONS_SCHEMA.empty_table()
    if "primary_key" not in points_table.column_names:
        raise ValueError(
            "evaluate_conformance_rules requires a 'primary_key' column on "
            "the points table",
        )

    chunks: list[pa.Table] = []
    for rule in pattern.conformance_rules:
        expr = compile_predicate(rule.violates_when)
        # ``filter`` evaluates ``expr`` row-wise; PyArrow handles compound
        # AND/OR/NOT natively without materializing intermediates.
        violators = points_table.filter(expr)
        n = violators.num_rows
        if n == 0:
            continue
        pk_arr = violators.column("primary_key").cast(pa.large_string())
        rule_arr = pa.array([rule.rule_id] * n, type=pa.large_string())
        sev_arr = pa.array([rule.severity] * n, type=pa.large_string())
        chunks.append(
            pa.Table.from_arrays(
                [pk_arr, rule_arr, sev_arr],
                schema=_VIOLATIONS_SCHEMA,
            ),
        )

    if not chunks:
        return _VIOLATIONS_SCHEMA.empty_table()
    return pa.concat_tables(chunks)


def conformance_dir(base_path: Path, pattern_id: str) -> Path:
    """Return the sidecar root directory for a pattern's violations.

    ``_gds_meta/conformance/violations/{pattern_id}/`` — version subdirs
    (``v={N}.lance``) and ``manifest.json`` live underneath.
    """
    return (
        base_path
        / "_gds_meta"
        / "conformance"
        / "violations"
        / pattern_id
    )


def conformance_dataset_path(
    base_path: Path, pattern_id: str, version: int,
) -> Path:
    """Return the Lance dataset path for a pattern's violations at a given
    geometry version.
    """
    return conformance_dir(base_path, pattern_id) / f"v={version}.lance"


def write_conformance_sidecar(
    base_path: Path,
    pattern_id: str,
    version: int,
    violations_table: pa.Table,
    rule_set_hash: str,
    n_rules: int,
) -> None:
    """Write the violations Lance dataset and adjacent manifest.

    The Lance dataset always carries ``_VIOLATIONS_SCHEMA``; when there
    are zero violations it is written as an empty dataset so that the
    bail-guard in the navigator (existence of the sphere-specific
    sidecar directory) remains a valid signal.
    """
    target_dir = conformance_dir(base_path, pattern_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Cast the incoming table to the canonical schema — defensive in case
    # an upstream call constructs primary_key/rule_id/severity as ``string``
    # rather than ``large_string``.
    if violations_table.schema != _VIOLATIONS_SCHEMA:
        violations_table = violations_table.cast(_VIOLATIONS_SCHEMA)

    dataset_path = conformance_dataset_path(base_path, pattern_id, version)
    _lance.write_dataset(violations_table, str(dataset_path), mode="overwrite")

    manifest = {
        "version": version,
        "rule_set_hash": rule_set_hash,
        "n_rules": n_rules,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def build_conformance_for_pattern(
    base_path: Path,
    pattern: Pattern,
    points_table: pa.Table,
    version: int,
) -> dict[str, Any] | None:
    """Evaluate and persist conformance for one pattern.

    Returns the manifest dict on success, ``None`` when the pattern has
    no rules (cost-neutral fast path).
    """
    if not pattern.conformance_rules:
        return None
    from hypertopos.model.sphere import compute_rule_set_hash

    violations = evaluate_conformance_rules(pattern, points_table)
    rule_set_hash = compute_rule_set_hash(pattern.conformance_rules)
    write_conformance_sidecar(
        base_path=base_path,
        pattern_id=pattern.pattern_id,
        version=version,
        violations_table=violations,
        rule_set_hash=rule_set_hash,
        n_rules=len(pattern.conformance_rules),
    )
    return {
        "version": version,
        "rule_set_hash": rule_set_hash,
        "n_rules": len(pattern.conformance_rules),
        "n_violations": violations.num_rows,
    }
