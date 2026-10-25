# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import threading
from collections import OrderedDict

from hypertopos.model.objects import Polygon


class GDSCache:
    """Polygon LRU shared by the sessions of one sphere.

    Entries are keyed by the reader snapshot they were built from, so a
    session never gets a polygon computed under another snapshot. Sessions and
    parallel tool steps use it from several threads, so every access holds a lock.
    """

    def __init__(self, max_polygons: int = 1000) -> None:
        self._polygons: OrderedDict[tuple[str, str, object], Polygon] = OrderedDict()
        self._max: int = max_polygons
        self._lock = threading.Lock()

    def get_polygon(
        self, primary_key: str, pattern_id: str, snapshot_id: object = 0
    ) -> Polygon | None:
        key = (primary_key, pattern_id, snapshot_id)
        with self._lock:
            polygon = self._polygons.get(key)
            if polygon is not None:
                self._polygons.move_to_end(key)
            return polygon

    def put_polygon(self, polygon: Polygon, snapshot_id: object = 0) -> None:
        key = (polygon.primary_key, polygon.pattern_id, snapshot_id)
        with self._lock:
            self._polygons[key] = polygon
            self._polygons.move_to_end(key)
            if len(self._polygons) > self._max:
                self._polygons.popitem(last=False)
