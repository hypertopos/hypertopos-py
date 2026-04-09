# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from hypertopos.model.objects import Polygon
from hypertopos.storage.cache import GDSCache


def _make_polygon(key: str) -> Polygon:
    return Polygon(
        primary_key=key,
        pattern_id="p",
        pattern_ver=1,
        pattern_type="anchor",
        scale=1,
        delta=np.zeros(2),
        delta_norm=0.0,
        is_anomaly=False,
        edges=[],
        last_refresh_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_cache_put_and_get():
    cache = GDSCache()
    p = _make_polygon("CUST-001")
    cache.put_polygon(p)
    result = cache.get_polygon("CUST-001", "p")
    assert result is not None
    assert result.primary_key == "CUST-001"


def test_cache_miss():
    cache = GDSCache()
    assert cache.get_polygon("MISSING", "p") is None
