# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""YAML loader for the per-pattern ``conformance_rules:`` block.

Translates the declarative YAML form into the ``ConformanceRule`` /
``ConformancePredicate`` ASTs consumed by the runtime evaluator
(``hypertopos.builder.conformance``). The YAML predicate shape mirrors
``hypertopos.model.sphere._predicate_to_dict`` exactly — same keys
(``op``, ``terms``, ``prop``, ``value``), so a round-trip
build → read is structurally lossless.

Validation
----------
Two layers, both pure-Python (no eval, no compile):

  1. Structural — every node carries a known ``op``; logical compounds
     carry non-empty ``terms``; comparisons carry ``prop`` + ``value``;
     ``not`` has exactly one child; ``in`` has an iterable ``value``.

  2. Column references (opt-in) — when the caller passes
     ``available_columns`` (the entity-line points-table column set),
     every leaf comparison's ``prop`` must appear in it. ``None``
     disables the check, which keeps the loader unit-testable without
     a builder.

Note: ``prop`` references columns on the **points table**, not pattern
``dim_labels``. The runtime compiler walks the AST and emits
``pyarrow.compute.field(prop)`` against the entity-line table — so
``prop`` is a raw source column name, not a dimension name.
"""
from __future__ import annotations

from typing import Any

from hypertopos.model.sphere import (
    ConformancePredicate,
    ConformanceRule,
)

# Logical compounds — drive the recursive walk on ``terms``.
_LOGICAL_OPS: frozenset[str] = frozenset({"and", "or", "not"})

# Comparison ops — leaf nodes carrying ``prop`` + ``value``.
_BINARY_COMPARISON_OPS: frozenset[str] = frozenset(
    {"==", "!=", "<", "<=", ">", ">="},
)

_VALID_SEVERITIES: frozenset[str] = frozenset(
    {"low", "medium", "high", "critical"},
)


def _parse_predicate(
    raw: Any,
    *,
    pattern_id: str,
    rule_id: str,
    available_columns: set[str] | None,
    path: str = "violates_when",
) -> ConformancePredicate:
    """Recursively parse a YAML predicate dict into a ``ConformancePredicate``.

    ``path`` is the dotted location used in error messages, e.g.
    ``violates_when.terms[0]``.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"Pattern '{pattern_id}' rule '{rule_id}' {path}: predicate "
            f"must be a YAML mapping; got {type(raw).__name__}",
        )
    op = raw.get("op")
    if not isinstance(op, str):
        raise ValueError(
            f"Pattern '{pattern_id}' rule '{rule_id}' {path}: predicate "
            f"missing string 'op'",
        )

    if op in _LOGICAL_OPS:
        terms_raw = raw.get("terms")
        if not isinstance(terms_raw, list) or not terms_raw:
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' {path}: "
                f"'{op}' predicate requires a non-empty 'terms' list",
            )
        if op == "not" and len(terms_raw) != 1:
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' {path}: "
                f"'not' predicate requires exactly one term; "
                f"got {len(terms_raw)}",
            )
        terms = [
            _parse_predicate(
                t,
                pattern_id=pattern_id,
                rule_id=rule_id,
                available_columns=available_columns,
                path=f"{path}.terms[{i}]",
            )
            for i, t in enumerate(terms_raw)
        ]
        return ConformancePredicate(op=op, terms=terms)  # type: ignore[arg-type]

    if op in _BINARY_COMPARISON_OPS or op == "in":
        prop = raw.get("prop")
        if not isinstance(prop, str) or not prop:
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' {path}: "
                f"'{op}' predicate requires a non-empty string 'prop'",
            )
        if "value" not in raw:
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' {path}: "
                f"'{op}' predicate requires a 'value' field",
            )
        value = raw["value"]
        if op == "in":
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"Pattern '{pattern_id}' rule '{rule_id}' {path}: "
                    f"'in' predicate requires an iterable 'value' "
                    f"(list/tuple); got {type(value).__name__}",
                )
            value = list(value)
        if available_columns is not None and prop not in available_columns:
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' {path}: "
                f"prop '{prop}' is not a column on the entity points "
                f"table. Available columns: {sorted(available_columns)}",
            )
        return ConformancePredicate(op=op, prop=prop, value=value)  # type: ignore[arg-type]

    raise ValueError(
        f"Pattern '{pattern_id}' rule '{rule_id}' {path}: "
        f"unknown 'op' {op!r}. Valid: "
        f"{sorted(_LOGICAL_OPS | _BINARY_COMPARISON_OPS | {'in'})}",
    )


def parse_conformance_rules(
    raw_block: Any,
    *,
    pattern_id: str,
    available_columns: set[str] | None = None,
) -> list[ConformanceRule]:
    """Parse a YAML ``conformance_rules:`` block into ``ConformanceRule``s.

    Args:
        raw_block: The value of the YAML ``conformance_rules:`` key under
            a pattern. Must be a list of rule dicts. ``None`` is treated
            as an empty list.
        pattern_id: For error messages.
        available_columns: Optional set of column names on the
            entity-line points table. When provided, every leaf
            comparison's ``prop`` must appear in it. ``None`` skips the
            check (keeps the loader unit-testable without a builder).

    Returns:
        List of ``ConformanceRule`` ready to be attached to the
        builder's ``_PatternReg.conformance_rules`` slot and consumed by
        the runtime evaluator.

    Rule auto-id: when an entry omits ``rule_id``, the loader assigns
    ``f"r{i}"`` (i = entry index, 0-based) so the rule_set hash stays
    stable across builds of the same YAML.

    Raises:
        ValueError on any schema violation — unknown op, missing prop /
        value, malformed terms, invalid severity, duplicate rule_id, or
        prop referring to a column not on the entity points table.
    """
    if raw_block is None:
        return []
    if not isinstance(raw_block, list):
        raise ValueError(
            f"Pattern '{pattern_id}' conformance_rules must be a YAML "
            f"list of rule entries; got {type(raw_block).__name__}",
        )

    rules: list[ConformanceRule] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw_block):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Pattern '{pattern_id}' conformance_rules[{i}] must be a "
                f"YAML mapping; got {type(entry).__name__}",
            )

        rule_id_raw = entry.get("rule_id")
        if rule_id_raw is None:
            rule_id = f"r{i}"
        elif isinstance(rule_id_raw, str) and rule_id_raw:
            rule_id = rule_id_raw
        else:
            raise ValueError(
                f"Pattern '{pattern_id}' conformance_rules[{i}].rule_id "
                f"must be a non-empty string when present",
            )
        if rule_id in seen_ids:
            raise ValueError(
                f"Pattern '{pattern_id}' conformance_rules: duplicate "
                f"rule_id {rule_id!r}",
            )
        seen_ids.add(rule_id)

        severity = entry.get("severity")
        if not isinstance(severity, str) or severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' severity must "
                f"be one of {sorted(_VALID_SEVERITIES)}; got "
                f"{severity!r}",
            )

        violates_when_raw = entry.get("violates_when")
        if violates_when_raw is None:
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' missing "
                f"'violates_when' predicate",
            )

        predicate = _parse_predicate(
            violates_when_raw,
            pattern_id=pattern_id,
            rule_id=rule_id,
            available_columns=available_columns,
        )

        description = entry.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(
                f"Pattern '{pattern_id}' rule '{rule_id}' description "
                f"must be a string when present; got "
                f"{type(description).__name__}",
            )

        rules.append(
            ConformanceRule(
                rule_id=rule_id,
                severity=severity,  # type: ignore[arg-type]
                violates_when=predicate,
                description=description,
            ),
        )

    return rules
