# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for the conformance YAML loader.

Covers:
  * leaf comparison + nested and/or/not predicate parsing
  * ``in`` predicate with list value
  * auto rule_id assignment when ``rule_id`` omitted
  * column-reference validation against the entity points table
  * malformed-input errors (unknown op, missing terms, bad severity,
    duplicate rule_id, non-list block, non-iterable ``in`` value, ...)
  * end-to-end YAML → builder → sidecar → runtime evaluator round-trip
"""
from __future__ import annotations

import json

import pyarrow as pa
import pytest
from hypertopos.builder.builder import GDSBuilder, RelationSpec
from hypertopos.builder.conformance import evaluate_conformance_rules
from hypertopos.builder.conformance_mapping import parse_conformance_rules
from hypertopos.engine.conformance import read_violations
from hypertopos.model.sphere import Pattern

# ---------------------------------------------------------------------------
# Unit tests — predicate / rule parsing
# ---------------------------------------------------------------------------


def test_parse_leaf_eq_predicate():
    """A single ``a == 1`` rule produces the expected AST."""
    raw = [
        {
            "rule_id": "r_a",
            "severity": "high",
            "violates_when": {"op": "==", "prop": "a", "value": 1},
        },
    ]
    rules = parse_conformance_rules(raw, pattern_id="p")
    assert len(rules) == 1
    rule = rules[0]
    assert rule.rule_id == "r_a"
    assert rule.severity == "high"
    assert rule.violates_when.op == "=="
    assert rule.violates_when.prop == "a"
    assert rule.violates_when.value == 1


def test_parse_nested_and_or_not_predicate():
    """``and(or(a==1, b==2), not(c<5))`` round-trips into the expected AST."""
    raw = [
        {
            "rule_id": "r_compound",
            "severity": "medium",
            "violates_when": {
                "op": "and",
                "terms": [
                    {
                        "op": "or",
                        "terms": [
                            {"op": "==", "prop": "a", "value": 1},
                            {"op": "==", "prop": "b", "value": 2},
                        ],
                    },
                    {
                        "op": "not",
                        "terms": [
                            {"op": "<", "prop": "c", "value": 5},
                        ],
                    },
                ],
            },
        },
    ]
    rules = parse_conformance_rules(raw, pattern_id="p")
    pred = rules[0].violates_when
    assert pred.op == "and"
    assert pred.terms is not None and len(pred.terms) == 2
    or_node, not_node = pred.terms
    assert or_node.op == "or"
    assert not_node.op == "not"
    assert or_node.terms is not None and len(or_node.terms) == 2
    assert not_node.terms is not None and len(not_node.terms) == 1
    assert not_node.terms[0].op == "<"
    assert not_node.terms[0].prop == "c"


def test_parse_in_predicate_with_list():
    """An ``in`` predicate accepts a list value and preserves order."""
    raw = [
        {
            "rule_id": "r_in",
            "severity": "low",
            "violates_when": {
                "op": "in",
                "prop": "region",
                "value": ["EU", "APAC"],
            },
        },
    ]
    rules = parse_conformance_rules(raw, pattern_id="p")
    pred = rules[0].violates_when
    assert pred.op == "in"
    assert pred.prop == "region"
    assert pred.value == ["EU", "APAC"]


def test_parse_all_comparison_ops():
    """Every supported comparison op parses without error."""
    ops = ["==", "!=", "<", "<=", ">", ">="]
    raw = [
        {
            "rule_id": f"r_{i}",
            "severity": "low",
            "violates_when": {"op": op, "prop": "x", "value": 0},
        }
        for i, op in enumerate(ops)
    ]
    rules = parse_conformance_rules(raw, pattern_id="p")
    assert [r.violates_when.op for r in rules] == ops


def test_auto_rule_id_when_omitted():
    """Rules without ``rule_id`` are auto-assigned ``r0``, ``r1``, ..."""
    raw = [
        {
            "severity": "high",
            "violates_when": {"op": "==", "prop": "a", "value": 1},
        },
        {
            "severity": "low",
            "violates_when": {"op": "==", "prop": "b", "value": 2},
        },
    ]
    rules = parse_conformance_rules(raw, pattern_id="p")
    assert [r.rule_id for r in rules] == ["r0", "r1"]


def test_description_optional():
    """Optional ``description`` field is preserved when present."""
    raw = [
        {
            "rule_id": "r_d",
            "severity": "critical",
            "description": "audit-trail required",
            "violates_when": {"op": ">", "prop": "amount", "value": 1000},
        },
    ]
    rules = parse_conformance_rules(raw, pattern_id="p")
    assert rules[0].description == "audit-trail required"


def test_none_block_returns_empty_list():
    """``conformance_rules: null`` (or absent block) yields an empty list."""
    assert parse_conformance_rules(None, pattern_id="p") == []


def test_empty_list_block_returns_empty_list():
    """An empty list block yields zero rules without error."""
    assert parse_conformance_rules([], pattern_id="p") == []


# ---------------------------------------------------------------------------
# Column-reference validation
# ---------------------------------------------------------------------------


def test_available_columns_accepts_known_prop():
    """Leaf ``prop`` referring to an existing column passes validation."""
    raw = [
        {
            "rule_id": "r_ok",
            "severity": "low",
            "violates_when": {"op": "==", "prop": "country", "value": "XX"},
        },
    ]
    parse_conformance_rules(
        raw, pattern_id="p", available_columns={"country", "amount"},
    )


def test_available_columns_rejects_unknown_prop():
    """Leaf ``prop`` not in the column set raises ValueError."""
    raw = [
        {
            "rule_id": "r_bad",
            "severity": "low",
            "violates_when": {"op": "==", "prop": "ghost_col", "value": 1},
        },
    ]
    with pytest.raises(ValueError, match=r"ghost_col.*not a column"):
        parse_conformance_rules(
            raw, pattern_id="p", available_columns={"a", "b"},
        )


def test_available_columns_none_skips_check():
    """``available_columns=None`` disables the prop-column check entirely."""
    raw = [
        {
            "rule_id": "r",
            "severity": "low",
            "violates_when": {"op": "==", "prop": "anything", "value": 1},
        },
    ]
    # No raise even though no columns are passed.
    parse_conformance_rules(raw, pattern_id="p", available_columns=None)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_op_raises():
    """An op outside the canonical set raises with the valid-op list."""
    raw = [
        {
            "rule_id": "r",
            "severity": "low",
            "violates_when": {"op": "between", "prop": "x", "value": [0, 1]},
        },
    ]
    with pytest.raises(ValueError, match=r"unknown 'op' 'between'"):
        parse_conformance_rules(raw, pattern_id="p")


def test_not_predicate_requires_single_term():
    """A ``not`` with !=1 term raises immediately."""
    raw = [
        {
            "rule_id": "r",
            "severity": "low",
            "violates_when": {
                "op": "not",
                "terms": [
                    {"op": "==", "prop": "a", "value": 1},
                    {"op": "==", "prop": "b", "value": 2},
                ],
            },
        },
    ]
    with pytest.raises(ValueError, match=r"'not' predicate requires exactly one term"):
        parse_conformance_rules(raw, pattern_id="p")


def test_logical_op_requires_nonempty_terms():
    """Empty ``terms`` on ``and`` raises with a helpful message."""
    raw = [
        {
            "rule_id": "r",
            "severity": "low",
            "violates_when": {"op": "and", "terms": []},
        },
    ]
    with pytest.raises(ValueError, match=r"requires a non-empty 'terms'"):
        parse_conformance_rules(raw, pattern_id="p")


def test_in_predicate_requires_iterable_value():
    """``in`` with a scalar ``value`` raises ValueError."""
    raw = [
        {
            "rule_id": "r",
            "severity": "low",
            "violates_when": {"op": "in", "prop": "region", "value": "EU"},
        },
    ]
    with pytest.raises(ValueError, match=r"'in' predicate requires an iterable"):
        parse_conformance_rules(raw, pattern_id="p")


def test_comparison_predicate_requires_prop():
    """A ``==`` predicate missing ``prop`` raises."""
    raw = [
        {
            "rule_id": "r",
            "severity": "low",
            "violates_when": {"op": "==", "value": 1},
        },
    ]
    with pytest.raises(ValueError, match=r"requires a non-empty string 'prop'"):
        parse_conformance_rules(raw, pattern_id="p")


def test_comparison_predicate_requires_value_field():
    """A comparison missing ``value`` raises (None is a legal value, so we
    check field presence, not truthiness)."""
    raw = [
        {
            "rule_id": "r",
            "severity": "low",
            "violates_when": {"op": "==", "prop": "a"},
        },
    ]
    with pytest.raises(ValueError, match=r"requires a 'value' field"):
        parse_conformance_rules(raw, pattern_id="p")


def test_invalid_severity_raises():
    """A severity outside {low, medium, high, critical} raises."""
    raw = [
        {
            "rule_id": "r",
            "severity": "blocker",  # not in the canonical set
            "violates_when": {"op": "==", "prop": "a", "value": 1},
        },
    ]
    with pytest.raises(ValueError, match=r"severity must be one of"):
        parse_conformance_rules(raw, pattern_id="p")


def test_duplicate_rule_id_raises():
    """Two rules with the same ``rule_id`` are rejected at parse time."""
    raw = [
        {
            "rule_id": "dup",
            "severity": "low",
            "violates_when": {"op": "==", "prop": "a", "value": 1},
        },
        {
            "rule_id": "dup",
            "severity": "high",
            "violates_when": {"op": "==", "prop": "b", "value": 2},
        },
    ]
    with pytest.raises(ValueError, match=r"duplicate rule_id 'dup'"):
        parse_conformance_rules(raw, pattern_id="p")


def test_non_list_block_raises():
    """A non-list top-level block is rejected (catches typos like a dict)."""
    raw = {"rule_id": "r", "severity": "low"}  # forgot the list wrapper
    with pytest.raises(ValueError, match=r"must be a YAML list"):
        parse_conformance_rules(raw, pattern_id="p")


def test_missing_violates_when_raises():
    """A rule entry without ``violates_when`` raises."""
    raw = [{"rule_id": "r", "severity": "low"}]
    with pytest.raises(ValueError, match=r"missing 'violates_when'"):
        parse_conformance_rules(raw, pattern_id="p")


def test_empty_rule_id_string_raises():
    """An empty string ``rule_id`` is treated as malformed."""
    raw = [
        {
            "rule_id": "",
            "severity": "low",
            "violates_when": {"op": "==", "prop": "a", "value": 1},
        },
    ]
    with pytest.raises(ValueError, match=r"rule_id must be a non-empty string"):
        parse_conformance_rules(raw, pattern_id="p")


# ---------------------------------------------------------------------------
# Integration — parsed rules consumed by the runtime evaluator
# ---------------------------------------------------------------------------


def _ten_row_table() -> pa.Table:
    """Tiny synthetic points table — used by the integration test."""
    return pa.table(
        {
            "primary_key": [f"E{i}" for i in range(10)],
            "amount": [200, 150, 110, 50, 40, 30, 20, 10, 5, 1],
            "region": [
                "EU", "XX", "US", "EU", "US", "EU", "US", "EU", "US", "EU",
            ],
        },
    )


def test_parsed_rules_match_handcoded_runtime_violations():
    """Parsing YAML and hand-building the same ASTs yield identical
    violations through ``evaluate_conformance_rules``."""
    raw = [
        {
            "rule_id": "rule_high",
            "severity": "high",
            "violates_when": {"op": ">", "prop": "amount", "value": 100},
        },
        {
            "rule_id": "rule_in",
            "severity": "low",
            "violates_when": {
                "op": "in", "prop": "region", "value": ["XX", "APAC"],
            },
        },
    ]
    parsed = parse_conformance_rules(
        raw,
        pattern_id="p_x",
        available_columns={"primary_key", "amount", "region"},
    )

    # Shim a minimal Pattern carrying just the rules — the runtime evaluator
    # only reads ``conformance_rules`` off the Pattern.
    pattern = Pattern.__new__(Pattern)
    object.__setattr__(pattern, "pattern_id", "p_x")
    object.__setattr__(pattern, "conformance_rules", parsed)

    violations = evaluate_conformance_rules(pattern, _ten_row_table())
    # ``amount > 100`` → E0, E1, E2 (3 rows); ``region in {XX, APAC}`` → E1 (1).
    assert violations.num_rows == 4
    rows = set(
        zip(
            violations.column("primary_key").to_pylist(),
            violations.column("rule_id").to_pylist(),
            strict=True,
        ),
    )
    assert ("E0", "rule_high") in rows
    assert ("E1", "rule_high") in rows
    assert ("E2", "rule_high") in rows
    assert ("E1", "rule_in") in rows


# ---------------------------------------------------------------------------
# End-to-end — YAML → cli/schema → builder → sidecar → read_violations
# ---------------------------------------------------------------------------


def test_yaml_pattern_config_carries_conformance_rules(tmp_path):
    """A sphere.yaml with two conformance rules surfaces them on
    ``PatternConfig.conformance_rules`` as a raw list (parsed lazily
    further down the pipeline)."""
    from hypertopos.cli.schema import parse_config

    yaml_text = """
version: "0.1.0"
sphere_id: test_sphere
sources:
  src1:
    path: data.csv
lines:
  accounts:
    source: src1
    key: primary_key
    role: anchor
patterns:
  account_pattern:
    type: anchor
    entity_line: accounts
    relations: []
    conformance_rules:
      - rule_id: high_amount
        severity: high
        description: amount exceeds compliance threshold
        violates_when:
          op: ">"
          prop: amount
          value: 100
      - rule_id: sanctioned_region
        severity: critical
        violates_when:
          op: in
          prop: region
          value: [XX, ZZ]
"""
    path = tmp_path / "sphere.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    cfg = parse_config(path)
    pat = cfg.patterns["account_pattern"]
    assert pat.conformance_rules is not None
    assert len(pat.conformance_rules) == 2
    assert pat.conformance_rules[0]["rule_id"] == "high_amount"
    assert pat.conformance_rules[1]["rule_id"] == "sanctioned_region"


def test_builder_end_to_end_writes_sidecar_with_yaml_rules(tmp_path):
    """Toy YAML → builder.add_pattern(conformance_rules=parsed) →
    sidecar Lance dataset on disk → ``read_violations`` returns the
    expected rows. Validates the full YAML-loader wiring without a sphere.yaml
    rebuild (which would be out of scope for a unit test)."""
    accounts = pa.table(
        {
            "primary_key": [f"A{i}" for i in range(6)],
            "amount": [200, 150, 110, 50, 40, 10],
            "region": ["EU", "XX", "US", "EU", "ZZ", "EU"],
        },
    )

    # Toy YAML block — exactly the shape that lands on PatternConfig.
    raw_rules = [
        {
            "rule_id": "high_amount",
            "severity": "high",
            "violates_when": {"op": ">", "prop": "amount", "value": 100},
        },
        {
            "rule_id": "sanctioned_region",
            "severity": "critical",
            "violates_when": {
                "op": "in", "prop": "region", "value": ["XX", "ZZ"],
            },
        },
    ]
    rules = parse_conformance_rules(
        raw_rules,
        pattern_id="account_pattern",
        available_columns=set(accounts.column_names),
    )

    out_path = tmp_path / "gds_toy"
    builder = GDSBuilder("toy", str(out_path))
    builder.add_line(
        "accounts",
        accounts,
        key_col="primary_key",
        source_id="accounts",
        role="anchor",
    )
    builder.add_pattern(
        "account_pattern",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[
            RelationSpec(
                line_id="accounts",
                fk_col=None,
                direction="self",
                required=True,
            ),
        ],
        conformance_rules=rules,
    )
    builder.build()

    # Sidecar landed under _gds_meta/conformance/violations/account_pattern.
    violations, manifest = read_violations(
        base_path=out_path,
        pattern_id="account_pattern",
        version=1,
        top_n=100,
    )
    assert manifest is not None
    assert manifest["n_rules"] == 2

    # ``amount > 100`` fires on A0, A1, A2 (3 rows).
    # ``region in {XX, ZZ}`` fires on A1, A4 (2 rows). Total 5.
    assert len(violations) == 5
    pks_by_rule: dict[str, set[str]] = {}
    for v in violations:
        pks_by_rule.setdefault(v["rule_id"], set()).add(v["primary_key"])
    assert pks_by_rule["high_amount"] == {"A0", "A1", "A2"}
    assert pks_by_rule["sanctioned_region"] == {"A1", "A4"}

    # sphere.json must carry the rules embedded under the pattern entry —
    # guards against silent regressions in the pattern serializer.
    sphere_json = json.loads(
        (out_path / "_gds_meta" / "sphere.json").read_text(),
    )
    pat_dict = sphere_json["patterns"]["account_pattern"]
    assert "conformance_rules" in pat_dict
    assert len(pat_dict["conformance_rules"]) == 2
    rule_ids = {r["rule_id"] for r in pat_dict["conformance_rules"]}
    assert rule_ids == {"high_amount", "sanctioned_region"}


def test_builder_no_rules_writes_no_sidecar(tmp_path):
    """A pattern without conformance_rules produces no sidecar directory —
    cost-neutral fast path verified end-to-end."""
    accounts = pa.table(
        {"primary_key": ["A0", "A1", "A2"], "amount": [10, 20, 30]},
    )
    out_path = tmp_path / "gds_no_rules"
    builder = GDSBuilder("toy_no_rules", str(out_path))
    builder.add_line(
        "accounts",
        accounts,
        key_col="primary_key",
        source_id="accounts",
        role="anchor",
    )
    builder.add_pattern(
        "account_pattern",
        pattern_type="anchor",
        entity_line="accounts",
        relations=[
            RelationSpec(
                line_id="accounts",
                fk_col=None,
                direction="self",
                required=True,
            ),
        ],
        # No conformance_rules argument.
    )
    builder.build()

    sidecar_dir = (
        out_path / "_gds_meta" / "conformance" / "violations" / "account_pattern"
    )
    assert not sidecar_dir.exists()
