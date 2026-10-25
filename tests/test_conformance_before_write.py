# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Regression test: conformance check must run BEFORE any point dataset is written.

If the conformance evaluation raises (e.g. unsupported predicate op), no Lance
files should be left on disk under ``points/``.
"""

from __future__ import annotations

import pytest

from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.model.sphere import ConformancePredicate, ConformanceRule


def _bad_rule() -> ConformanceRule:
    """A conformance rule whose predicate uses an unsupported op.

    ``compile_predicate`` raises ``ValueError`` for unknown ops, so
    ``_evaluate_conformance_rules`` propagates that error before (or, after
    the fix, instead of) writing points to disk.
    """
    return ConformanceRule(
        rule_id="bad_rule",
        severity="critical",
        violates_when=ConformancePredicate(op="INVALID_OP", prop="amount", value=0),
    )


def _build_with_bad_rule(tmp_path):
    """Minimal anchor sphere whose sole conformance rule is malformed."""
    b = GDSBuilder("cf_test", str(tmp_path / "gds_cf"))
    b.add_line(
        "items",
        [
            {"item_id": "I-1", "amount": 10},
            {"item_id": "I-2", "amount": 20},
            {"item_id": "I-3", "amount": 30},
        ],
        key_col="item_id",
        source_id="test",
    )
    b.add_pattern(
        "item_pattern",
        pattern_type="anchor",
        entity_line="items",
        relations=[RelationSpec(line_id="items", fk_col=None, direction="self")],
        conformance_rules=[_bad_rule()],
    )
    return b


def test_conformance_raises_before_points_written(tmp_path):
    """Build must raise from the malformed conformance rule AND leave no
    point datasets on disk.

    On unfixed code the build writes ``points/items/v=1.lance`` first and
    then raises — so ``points/`` exists after the exception.  After the fix
    the conformance check runs before any write, so ``points/`` must be absent.
    """
    b = _build_with_bad_rule(tmp_path)
    output_dir = tmp_path / "gds_cf"

    with pytest.raises(ValueError, match="INVALID_OP"):
        b.build()

    points_dir = output_dir / "points"
    assert not points_dir.exists(), (
        "points/ directory must not exist when conformance check fires before write; "
        f"found: {list(points_dir.iterdir()) if points_dir.exists() else 'N/A'}"
    )
