"""GDSCache is shared by sessions and by parallel detect_pattern steps: it must be thread-safe."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from unittest.mock import MagicMock

from hypertopos.storage.cache import GDSCache


def _polygon(key: str) -> MagicMock:
    poly = MagicMock()
    poly.primary_key = key
    poly.pattern_id = "p"
    return poly


class _SlowMoveDict(OrderedDict):
    """Widens the check-then-move window so a missing lock shows up at once."""

    def move_to_end(self, key, last=True):  # type: ignore[no-untyped-def]
        time.sleep(0.0005)
        super().move_to_end(key, last)


def test_concurrent_get_and_put_never_raise() -> None:
    cache = GDSCache(max_polygons=2)
    cache._polygons = _SlowMoveDict()
    cache.put_polygon(_polygon("hot"), snapshot_id=1)
    errors: list[BaseException] = []
    stop = time.monotonic() + 3.0

    def reader() -> None:
        try:
            while time.monotonic() < stop and not errors:
                cache.get_polygon("hot", "p", 1)
        except BaseException as exc:  # noqa: BLE001 - surface any failure
            errors.append(exc)

    def writer(prefix: str) -> None:
        try:
            i = 0
            while time.monotonic() < stop and not errors:
                cache.put_polygon(_polygon(f"{prefix}{i}"), snapshot_id=1)
                cache.put_polygon(_polygon("hot"), snapshot_id=1)
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=reader),
        threading.Thread(target=writer, args=("a",)),
        threading.Thread(target=writer, args=("b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
