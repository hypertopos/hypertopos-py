# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for kind auto-detection functions in _bregman.py."""
import numpy as np
import pytest

from hypertopos.builder._bregman import (
    detect_kind_for_column,
    detect_kinds_for_pattern,
    format_kinds_summary,
)


class TestDetectKindForColumn:
    def test_binary_01_is_bernoulli(self):
        values = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        assert detect_kind_for_column(values) == "bernoulli"

    def test_nonneg_integers_is_poisson(self):
        values = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        assert detect_kind_for_column(values) == "poisson"

    def test_continuous_is_gaussian(self):
        values = np.array([1.5, 2.7, 3.14, 0.5])
        assert detect_kind_for_column(values) == "gaussian"

    def test_negative_values_is_gaussian(self):
        values = np.array([-1.0, 0.0, 1.0, 2.0])
        assert detect_kind_for_column(values) == "gaussian"

    def test_empty_finite_is_gaussian(self):
        values = np.array([np.nan, np.inf, -np.inf])
        assert detect_kind_for_column(values) == "gaussian"

    def test_mixed_float_int_is_gaussian(self):
        # [0.0, 1.0, 2.5] — contains non-integer float, so not bernoulli or poisson
        values = np.array([0.0, 1.0, 2.5])
        assert detect_kind_for_column(values) == "gaussian"

    def test_single_zero_is_bernoulli(self):
        # All values in {0.0, 1.0} — single zero still qualifies
        values = np.array([0.0, 0.0, 0.0])
        assert detect_kind_for_column(values) == "bernoulli"

    def test_single_one_is_bernoulli(self):
        values = np.array([1.0, 1.0, 1.0])
        assert detect_kind_for_column(values) == "bernoulli"

    def test_large_integers_is_poisson(self):
        values = np.array([100.0, 200.0, 300.0])
        assert detect_kind_for_column(values) == "poisson"


class TestDetectKindsForPattern:
    def test_binary_relations_bernoulli(self):
        # edge_max=None → bernoulli
        relation_edge_maxes = [None, None]
        derived_metrics = []
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["bernoulli", "bernoulli"]

    def test_edgemax_relations_poisson(self):
        # edge_max not None → poisson
        relation_edge_maxes = [5, 10]
        derived_metrics = []
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["poisson", "poisson"]

    def test_count_derived_poisson(self):
        relation_edge_maxes = []
        derived_metrics = ["count"]
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["poisson"]

    def test_sum_derived_gaussian(self):
        relation_edge_maxes = []
        derived_metrics = ["sum:amount"]
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["gaussian"]

    def test_windowed_count_derived_poisson(self):
        # "count:window=1d:agg=max" → base metric = "count" → poisson
        relation_edge_maxes = []
        derived_metrics = ["count:window=1d:agg=max"]
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["poisson"]

    def test_count_distinct_derived_poisson(self):
        relation_edge_maxes = []
        derived_metrics = ["count_distinct"]
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["poisson"]

    def test_precomputed_edgemax1_bernoulli(self):
        # edge_max == 1 → bernoulli
        relation_edge_maxes = []
        derived_metrics = []
        precomputed_edge_maxes = [1]
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["bernoulli"]

    def test_precomputed_edgemax_gt1_gaussian(self):
        # edge_max != 1 → gaussian
        relation_edge_maxes = []
        derived_metrics = []
        precomputed_edge_maxes = [5]
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["gaussian"]

    def test_prop_fill_bernoulli(self):
        relation_edge_maxes = []
        derived_metrics = []
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=3,
            dim_block_count=0,
        )
        assert result == ["bernoulli", "bernoulli", "bernoulli"]

    def test_dim_blocks_gaussian(self):
        relation_edge_maxes = []
        derived_metrics = []
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=4,
        )
        assert result == ["gaussian", "gaussian", "gaussian", "gaussian"]

    def test_override_takes_precedence(self):
        # derived_metrics = ["count"] would normally be poisson,
        # but override forces gaussian
        relation_edge_maxes = []
        derived_metrics = ["count", "sum:amount"]
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
            derived_kind_overrides=["gaussian", None],  # override first only
        )
        assert result == ["gaussian", "gaussian"]  # sum:amount still gaussian

    def test_override_none_uses_auto(self):
        # None override → auto detect
        relation_edge_maxes = [None]
        derived_metrics = []
        precomputed_edge_maxes = []
        result = detect_kinds_for_pattern(
            relation_edge_maxes=relation_edge_maxes,
            derived_metrics=derived_metrics,
            precomputed_edge_maxes=precomputed_edge_maxes,
            prop_count=0,
            dim_block_count=0,
        )
        assert result == ["bernoulli"]

    def test_full_mixed_pattern(self):
        # relations: [None, 5] → [bernoulli, poisson]
        # event dims: [binary_col, count_col] → [bernoulli, poisson]
        # derived: ["count", "sum:amount"] → [poisson, gaussian]
        # precomputed: [1, 3] → [bernoulli, gaussian]
        # prop_fill: 2 → [bernoulli, bernoulli]
        # dim_blocks: 1 → [gaussian]
        binary_col = np.array([0.0, 1.0, 0.0, 1.0])
        count_col = np.array([1.0, 2.0, 3.0, 4.0])

        result = detect_kinds_for_pattern(
            relation_edge_maxes=[None, 5],
            derived_metrics=["count", "sum:amount"],
            precomputed_edge_maxes=[1, 3],
            prop_count=2,
            dim_block_count=1,
            event_column_values=[binary_col, count_col],
        )
        expected = [
            "bernoulli",   # relation None
            "poisson",     # relation 5
            "bernoulli",   # event binary_col
            "poisson",     # event count_col
            "poisson",     # derived count
            "gaussian",    # derived sum:amount
            "bernoulli",   # precomputed edge_max=1
            "gaussian",    # precomputed edge_max=3
            "bernoulli",   # prop_fill
            "bernoulli",   # prop_fill
            "gaussian",    # dim_block
        ]
        assert result == expected


class TestFormatKindsSummary:
    def test_format(self):
        kinds = ["bernoulli"] * 4 + ["poisson"] * 2 + ["gaussian"] * 8
        result = format_kinds_summary(kinds)
        assert result == "bernoulli x4, poisson x2, gaussian x8"

    def test_format_single_kind(self):
        kinds = ["gaussian"] * 3
        result = format_kinds_summary(kinds)
        assert result == "gaussian x3"

    def test_format_empty(self):
        result = format_kinds_summary([])
        assert result == ""

    def test_format_preserves_order(self):
        # Should output kinds in the order they first appear
        kinds = ["gaussian", "bernoulli", "gaussian", "bernoulli"]
        result = format_kinds_summary(kinds)
        assert result == "gaussian x2, bernoulli x2"
