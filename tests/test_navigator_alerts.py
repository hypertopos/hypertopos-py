# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for check_alerts regime_changepoint false-positive filtering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypertopos.navigation.navigator import GDSNavigator


@pytest.fixture
def navigator_with_warning():
    """Navigator where pi12 returns a warning dict (no temporal data)."""
    nav = MagicMock(spec=GDSNavigator)
    nav.π12_attract_regime_change.return_value = [
        {"warning": "no temporal data found for the given range"}
    ]
    nav._check_regime_changepoint = GDSNavigator._check_regime_changepoint.__get__(nav)
    return nav


@pytest.fixture
def navigator_with_real_changepoint():
    """Navigator where pi12 returns a real changepoint."""
    nav = MagicMock(spec=GDSNavigator)
    nav.π12_attract_regime_change.return_value = [
        {"timestamp": "2026-01-15", "magnitude": 0.42, "dim": "transactions"}
    ]
    nav._check_regime_changepoint = GDSNavigator._check_regime_changepoint.__get__(nav)
    return nav


class TestRegimeChangepointFilter:
    def test_warning_dict_not_reported_as_alert(self, navigator_with_warning):
        """When pi12 returns a warning (no temporal data), no HIGH alert is emitted."""
        result = navigator_with_warning._check_regime_changepoint("test_pattern", {})
        assert result == [], f"Warning dict should produce no alerts, got: {result}"

    def test_real_changepoint_reported(self, navigator_with_real_changepoint):
        """Real changepoints must produce a HIGH alert."""
        result = navigator_with_real_changepoint._check_regime_changepoint("test_pattern", {})
        assert len(result) == 1
        assert result[0]["severity"] == "HIGH"
        assert result[0]["check_type"] == "regime_changepoint"
        assert "1 regime changepoint" in result[0]["message"]
