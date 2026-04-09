# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for cross-pattern composition via Fisher's method."""

import numpy as np
import pytest
from hypertopos.engine.composition import co_dispersion, fisher_combine_pvalues


class TestFisherCombine:
    def test_all_significant(self):
        p_values = [0.01, 0.02, 0.03]
        result = fisher_combine_pvalues(p_values)
        assert result["combined_p"] < 0.001
        assert result["chi2"] > 20
        assert result["df"] == 6

    def test_all_insignificant(self):
        p_values = [0.8, 0.7, 0.9]
        result = fisher_combine_pvalues(p_values)
        assert result["combined_p"] > 0.5

    def test_mixed_signals(self):
        p_values = [0.01, 0.5, 0.9]
        result = fisher_combine_pvalues(p_values)
        assert 0.01 < result["combined_p"] < 0.5

    def test_single_pattern(self):
        result = fisher_combine_pvalues([0.05])
        assert abs(result["combined_p"] - 0.05) < 0.02

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fisher_combine_pvalues([])

    def test_zero_pvalue_clamped(self):
        result = fisher_combine_pvalues([0.0, 0.5])
        assert result["combined_p"] < 0.01


class TestCoDispersion:
    def test_perfectly_correlated(self):
        norms_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norms_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = co_dispersion(norms_a, norms_b)
        assert result["spearman_rho"] > 0.9
        assert result["insufficient_data"] is False

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        norms_a = rng.random(100)
        norms_b = rng.random(100)
        result = co_dispersion(norms_a, norms_b)
        assert abs(result["spearman_rho"]) < 0.3

    def test_too_few_entities(self):
        result = co_dispersion(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert result["insufficient_data"] is True
