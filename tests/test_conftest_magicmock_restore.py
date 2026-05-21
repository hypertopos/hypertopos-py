# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Regression test for the conftest MagicMock-leak guard.

Scenario the autouse ``_restore_navigator_engine_class_attrs`` fixture in
``conftest.py`` defends against:

1. A test replaces a ``GDSNavigator`` class-level method with a ``MagicMock``
   via raw ``setattr`` (NOT ``monkeypatch.setattr``).
2. The test raises inside the patched context, short-circuiting any restore
   call placed after the ``setattr``.
3. Without the fixture, the next test that touches that method sees the
   ``MagicMock`` instead of the real function — silently masking regressions.

The two tests below must execute in file order: pytest's default collection
runs functions top-to-bottom, and this repo has no ``pytest-randomly`` /
``pytest-order`` plugin (verified against ``pyproject.toml``), so the order
is stable.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from hypertopos.navigation.navigator import GDSNavigator


def test_a_replaces_navigator_method_with_magicmock_then_raises() -> None:
    """Phase 1 — install a MagicMock at the class level and short-circuit."""
    setattr(GDSNavigator, "anomaly_summary", MagicMock(return_value=None))
    with pytest.raises(RuntimeError):
        raise RuntimeError("simulate a test failure inside the patched context")


def test_b_subsequent_test_sees_real_method() -> None:
    """Phase 2 — without the conftest fixture this assertion fails."""
    cls_attr = GDSNavigator.__dict__["anomaly_summary"]
    assert not isinstance(cls_attr, MagicMock), (
        "Class-level GDSNavigator.anomaly_summary leaked as MagicMock from a "
        "previous test — the conftest restore fixture did not run or did not "
        "detect the swap."
    )
    assert inspect.isfunction(cls_attr), (
        "Class-level GDSNavigator.anomaly_summary should be a plain function "
        f"after restore; got {type(cls_attr).__name__}."
    )
