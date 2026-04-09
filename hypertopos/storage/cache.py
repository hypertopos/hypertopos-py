# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from collections import OrderedDict

from hypertopos.model.objects import Polygon


class GDSCache:
    def __init__(self, max_polygons: int = 1000) -> None:
        self._polygons: OrderedDict[tuple[str, str], Polygon] = OrderedDict()
        self._max: int = max_polygons

    def get_polygon(self, primary_key: str, pattern_id: str) -> Polygon | None:
        key = (primary_key, pattern_id)
        if key not in self._polygons:
            return None
        self._polygons.move_to_end(key)
        return self._polygons[key]

    def put_polygon(self, polygon: Polygon) -> None:
        key = (polygon.primary_key, polygon.pattern_id)
        self._polygons[key] = polygon
        self._polygons.move_to_end(key)
        if len(self._polygons) > self._max:
            self._polygons.popitem(last=False)

