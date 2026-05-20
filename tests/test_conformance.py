# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for the conformance-rules subsystem (M1.7).

Covers:
  * predicate AST compilation (leaf eq, nested and/or/not, all comparisons)
  * vectorized evaluation against a points table
  * sidecar Lance dataset + manifest round-trip
  * read-time filter pushdown (rule_id, severity_min, top_n)
  * rule-set hash determinism and order invariance
  * navigator-level hash-mismatch warning
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
from hypertopos.builder.conformance import (
    compile_predicate,
    evaluate_conformance_rules,
    write_conformance_sidecar,
)
from hypertopos.engine.conformance import read_violations
from hypertopos.model.sphere import (
    ConformancePredicate,
    ConformanceRule,
    Pattern,
    RelationDef,
    Sphere,
    compute_rule_set_hash,
)
from hypertopos.navigation.navigator import GDSNavigator

_DT = datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _five_row_table() -> pa.Table:
    """5-entity points table with engineered values for predicate tests."""
    return pa.table(
        {
            "primary_key": ["E1", "E2", "E3", "E4", "E5"],
            "a": [1, 2, 3, 1, 5],
            "b": [9, 2, 7, 4, 2],
            "c": [10, 3, 6, 1, 8],
            "region": ["EU", "US", "EU", "APAC", "US"],
        },
    )


def _ten_row_table() -> pa.Table:
    """10-entity table for the multi-rule evaluation test.

    Designed so that rule_high (``amount > 100``) fires on rows 0, 1, 2 and
    rule_low (``region == 'XX'``) fires on row 1 — total of 3 distinct
    violation rows, with row 1 appearing twice.
    """
    return pa.table(
        {
            "primary_key": [f"E{i}" for i in range(10)],
            "amount": [200, 150, 110, 50, 40, 30, 20, 10, 5, 1],
            "region": ["EU", "XX", "US", "EU", "US", "EU", "US", "EU", "US", "EU"],
        },
    )


def _build_pattern(
    *,
    pattern_id: str = "p_x",
    conformance_rules: list[ConformanceRule] | None = None,
) -> Pattern:
    relations = [RelationDef(line_id="rel0", direction="in", required=True)]
    return Pattern(
        pattern_id=pattern_id,
        entity_type="x",
        pattern_type="anchor",
        relations=relations,
        mu=np.zeros(1, dtype=np.float32),
        sigma_diag=np.ones(1, dtype=np.float32),
        theta=np.ones(1, dtype=np.float32) * 3.0,
        population_size=10,
        computed_at=_DT,
        version=1,
        status="production",
        conformance_rules=conformance_rules or [],
    )


# ---------------------------------------------------------------------------
# Predicate compiler
# ---------------------------------------------------------------------------


def test_predicate_compiler_handles_leaf_eq():
    """A single ``a == 1`` predicate matches the two rows whose ``a`` is 1."""
    pred = ConformancePredicate(op="==", prop="a", value=1)
    expr = compile_predicate(pred)
    filtered = _five_row_table().filter(expr)
    pks = filtered.column("primary_key").to_pylist()
    assert sorted(pks) == ["E1", "E4"]


def test_predicate_compiler_handles_nested_and_or_not():
    """ADVISOR LOCK: ``and(or(a==1, b==2), not(c<5))`` truth-table check.

    Per-row truth values on the engineered 5-row table:
      E1: (a=1, b=9, c=10) → or(T,F)=T, not(10<5)=T → T  ✓
      E2: (a=2, b=2, c=3)  → or(F,T)=T, not(3<5)=F → F
      E3: (a=3, b=7, c=6)  → or(F,F)=F                  → F
      E4: (a=1, b=4, c=1)  → or(T,F)=T, not(1<5)=F → F
      E5: (a=5, b=2, c=8)  → or(F,T)=T, not(8<5)=T → T  ✓

    Only E1 and E5 should match.
    """
    pred = ConformancePredicate(
        op="and",
        terms=[
            ConformancePredicate(
                op="or",
                terms=[
                    ConformancePredicate(op="==", prop="a", value=1),
                    ConformancePredicate(op="==", prop="b", value=2),
                ],
            ),
            ConformancePredicate(
                op="not",
                terms=[
                    ConformancePredicate(op="<", prop="c", value=5),
                ],
            ),
        ],
    )
    expr = compile_predicate(pred)
    filtered = _five_row_table().filter(expr)
    pks = sorted(filtered.column("primary_key").to_pylist())
    assert pks == ["E1", "E5"], (
        f"hand-computed truth: only E1, E5 satisfy; got {pks}"
    )


def test_predicate_compiler_handles_comparison_ops():
    """Each comparison operator filters the expected subset of rows."""
    tbl = _five_row_table()

    # !=  → all rows where a is not 1 → E2, E3, E5
    expr = compile_predicate(ConformancePredicate(op="!=", prop="a", value=1))
    assert sorted(tbl.filter(expr).column("primary_key").to_pylist()) == [
        "E2", "E3", "E5",
    ]

    # <  → a < 3 → E1, E2, E4
    expr = compile_predicate(ConformancePredicate(op="<", prop="a", value=3))
    assert sorted(tbl.filter(expr).column("primary_key").to_pylist()) == [
        "E1", "E2", "E4",
    ]

    # <=  → a <= 2 → E1, E2, E4
    expr = compile_predicate(ConformancePredicate(op="<=", prop="a", value=2))
    assert sorted(tbl.filter(expr).column("primary_key").to_pylist()) == [
        "E1", "E2", "E4",
    ]

    # >  → c > 6 → E1, E5
    expr = compile_predicate(ConformancePredicate(op=">", prop="c", value=6))
    assert sorted(tbl.filter(expr).column("primary_key").to_pylist()) == [
        "E1", "E5",
    ]

    # >=  → c >= 6 → E1, E3, E5
    expr = compile_predicate(ConformancePredicate(op=">=", prop="c", value=6))
    assert sorted(tbl.filter(expr).column("primary_key").to_pylist()) == [
        "E1", "E3", "E5",
    ]

    # in  → region in {EU, APAC} → E1, E3, E4
    expr = compile_predicate(
        ConformancePredicate(op="in", prop="region", value=["EU", "APAC"]),
    )
    assert sorted(tbl.filter(expr).column("primary_key").to_pylist()) == [
        "E1", "E3", "E4",
    ]


# ---------------------------------------------------------------------------
# evaluate_conformance_rules
# ---------------------------------------------------------------------------


def test_evaluate_conformance_rules_returns_violations():
    """Two rules over 10 entities yield the expected 4 violation rows.

    rule_high (``amount > 100``) fires on E0, E1, E2.
    rule_low  (``region == 'XX'``) fires on E1.
    Total: 4 violation rows (E1 appears twice — once per rule).
    """
    rules = [
        ConformanceRule(
            rule_id="rule_high",
            severity="high",
            violates_when=ConformancePredicate(op=">", prop="amount", value=100),
        ),
        ConformanceRule(
            rule_id="rule_low",
            severity="low",
            violates_when=ConformancePredicate(
                op="==", prop="region", value="XX",
            ),
        ),
    ]
    pattern = _build_pattern(conformance_rules=rules)
    violations = evaluate_conformance_rules(pattern, _ten_row_table())
    assert violations.num_rows == 4
    rows = list(
        zip(
            violations.column("primary_key").to_pylist(),
            violations.column("rule_id").to_pylist(),
            violations.column("severity").to_pylist(),
            strict=True,
        ),
    )
    assert ("E0", "rule_high", "high") in rows
    assert ("E1", "rule_high", "high") in rows
    assert ("E2", "rule_high", "high") in rows
    assert ("E1", "rule_low", "low") in rows


def test_no_rules_returns_empty_table():
    """A pattern with zero rules produces an empty (schema-correct) table."""
    pattern = _build_pattern(conformance_rules=[])
    out = evaluate_conformance_rules(pattern, _ten_row_table())
    assert out.num_rows == 0
    assert out.column_names == ["primary_key", "rule_id", "severity"]


# ---------------------------------------------------------------------------
# Sidecar + manifest
# ---------------------------------------------------------------------------


def test_sidecar_manifest_carries_rule_set_hash(tmp_path: Path):
    """Manifest written by the builder carries the canonical rule_set_hash."""
    rules = [
        ConformanceRule(
            rule_id="r1",
            severity="medium",
            violates_when=ConformancePredicate(op=">", prop="amount", value=100),
        ),
    ]
    pattern = _build_pattern(conformance_rules=rules)
    violations = evaluate_conformance_rules(pattern, _ten_row_table())
    expected_hash = compute_rule_set_hash(rules)

    write_conformance_sidecar(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        violations_table=violations,
        rule_set_hash=expected_hash,
        n_rules=len(rules),
    )

    _, manifest = read_violations(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
    )
    assert manifest is not None
    assert manifest["rule_set_hash"] == expected_hash
    assert manifest["n_rules"] == 1
    assert manifest["version"] == 1
    assert "evaluated_at" in manifest


def test_read_violations_filters_by_rule_id(tmp_path: Path):
    """``rule_id=`` filter returns only that rule's violation rows."""
    rules = [
        ConformanceRule(
            rule_id="rule_high",
            severity="high",
            violates_when=ConformancePredicate(op=">", prop="amount", value=100),
        ),
        ConformanceRule(
            rule_id="rule_low",
            severity="low",
            violates_when=ConformancePredicate(
                op="==", prop="region", value="XX",
            ),
        ),
    ]
    pattern = _build_pattern(conformance_rules=rules)
    violations = evaluate_conformance_rules(pattern, _ten_row_table())
    write_conformance_sidecar(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        violations_table=violations,
        rule_set_hash=compute_rule_set_hash(rules),
        n_rules=len(rules),
    )

    out, _ = read_violations(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        rule_id="rule_low",
    )
    assert len(out) == 1
    assert out[0]["primary_key"] == "E1"
    assert out[0]["rule_id"] == "rule_low"


def test_read_violations_filters_by_severity_min(tmp_path: Path):
    """``severity_min='high'`` drops low-severity violations."""
    rules = [
        ConformanceRule(
            rule_id="rule_high",
            severity="high",
            violates_when=ConformancePredicate(op=">", prop="amount", value=100),
        ),
        ConformanceRule(
            rule_id="rule_low",
            severity="low",
            violates_when=ConformancePredicate(
                op="==", prop="region", value="XX",
            ),
        ),
    ]
    pattern = _build_pattern(conformance_rules=rules)
    violations = evaluate_conformance_rules(pattern, _ten_row_table())
    write_conformance_sidecar(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        violations_table=violations,
        rule_set_hash=compute_rule_set_hash(rules),
        n_rules=len(rules),
    )

    out, _ = read_violations(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        severity_min="high",
    )
    severities = {v["severity"] for v in out}
    assert severities == {"high"}, severities
    assert all(v["rule_id"] == "rule_high" for v in out)


def test_read_violations_top_n_caps(tmp_path: Path):
    """``top_n=K`` truncates the result to exactly min(N, K) rows."""
    rules = [
        ConformanceRule(
            rule_id="rule_high",
            severity="high",
            violates_when=ConformancePredicate(op=">", prop="amount", value=100),
        ),
    ]
    pattern = _build_pattern(conformance_rules=rules)
    violations = evaluate_conformance_rules(pattern, _ten_row_table())
    write_conformance_sidecar(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        violations_table=violations,
        rule_set_hash=compute_rule_set_hash(rules),
        n_rules=len(rules),
    )

    # Underlying dataset has 3 violations (E0, E1, E2).
    out_2, _ = read_violations(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        top_n=2,
    )
    assert len(out_2) == 2

    out_10, _ = read_violations(
        base_path=tmp_path,
        pattern_id=pattern.pattern_id,
        version=1,
        top_n=10,
    )
    assert len(out_10) == 3  # min(N=3, K=10)


# ---------------------------------------------------------------------------
# Navigator-level hash mismatch
# ---------------------------------------------------------------------------


def test_find_conformance_violations_warns_on_rule_set_hash_mismatch(
    tmp_path: Path,
):
    """Sidecar built with R1, navigator opens with R2 → warning, no raise."""
    r1 = ConformanceRule(
        rule_id="rule_a",
        severity="high",
        violates_when=ConformancePredicate(op=">", prop="amount", value=100),
    )
    r2 = ConformanceRule(
        rule_id="rule_b",  # different rule_id → different hash
        severity="low",
        violates_when=ConformancePredicate(
            op="==", prop="region", value="XX",
        ),
    )

    # Write sidecar for R1.
    pattern_r1 = _build_pattern(conformance_rules=[r1])
    violations = evaluate_conformance_rules(pattern_r1, _ten_row_table())
    write_conformance_sidecar(
        base_path=tmp_path,
        pattern_id=pattern_r1.pattern_id,
        version=1,
        violations_table=violations,
        rule_set_hash=compute_rule_set_hash([r1]),
        n_rules=1,
    )

    # Navigator sees a pattern declaring R2 instead.
    pattern_r2 = _build_pattern(conformance_rules=[r2])
    sphere = Sphere(
        sphere_id="s_x",
        name="s_x",
        base_path=str(tmp_path),
        patterns={pattern_r2.pattern_id: pattern_r2},
    )

    storage = MagicMock()
    storage._base = tmp_path
    storage.read_sphere.return_value = sphere
    manifest = MagicMock()
    manifest.pattern_version.return_value = 1

    nav = GDSNavigator.__new__(GDSNavigator)
    nav._storage = storage
    nav._manifest = manifest

    result = nav.find_conformance_violations(pattern_r2.pattern_id)
    assert any("rule_set_hash_mismatch" in w for w in result["warnings"]), (
        f"expected hash-mismatch warning; got warnings={result['warnings']!r}"
    )
    # Does NOT raise — sidecar still readable, violations from R1 surface.
    assert result["manifest"] is not None
    assert "violations" in result


def test_compute_rule_set_hash_deterministic_across_rule_order():
    """``hash([r1, r2]) == hash([r2, r1])`` (sort-by-rule_id is canonical)."""
    r1 = ConformanceRule(
        rule_id="alpha",
        severity="high",
        violates_when=ConformancePredicate(op=">", prop="amount", value=100),
    )
    r2 = ConformanceRule(
        rule_id="beta",
        severity="low",
        violates_when=ConformancePredicate(
            op="==", prop="region", value="XX",
        ),
    )
    h_ab = compute_rule_set_hash([r1, r2])
    h_ba = compute_rule_set_hash([r2, r1])
    assert h_ab == h_ba
    # Hash is deterministic across invocations.
    assert h_ab == compute_rule_set_hash([r1, r2])
    # Sanity check — a different rule set produces a different hash.
    r3 = ConformanceRule(
        rule_id="gamma",
        severity="medium",
        violates_when=ConformancePredicate(op="<", prop="amount", value=50),
    )
    assert compute_rule_set_hash([r1, r3]) != h_ab
