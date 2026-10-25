"""passive_scan output does not depend on which source thread finishes first."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from hypertopos.navigation.scanner import PassiveScanner, ScanSourceHit


def _hit(intensity: float) -> ScanSourceHit:
    return ScanSourceHit(
        anomalous_count=1, related_count=1, max_delta_norm=1.0, anomaly_intensity=intensity
    )


def test_hits_and_sources_ignore_thread_completion_order(monkeypatch) -> None:
    scanner = PassiveScanner(reader=MagicMock(), sphere=MagicMock(), manifest=MagicMock())
    monkeypatch.setattr(scanner, "_resolve_version", lambda _pid: 1)
    scanner._reader.count_points_rows.return_value = 10
    scanner._reader.read_points_schema.return_value = MagicMock()

    per_source = {
        "slow": {"K3": _hit(0.5), "K1": _hit(0.5), "K2": _hit(0.5)},
        "fast": {"K2": _hit(0.5), "K3": _hit(0.5), "K1": _hit(0.5)},
    }

    def fake_direct(source, _version):
        if source.name == "slow":
            time.sleep(0.2)
        return per_source[source.name]

    monkeypatch.setattr(scanner, "_scan_direct", fake_direct)
    scanner.add_source("slow", "p_slow", key_type="direct")
    scanner.add_source("fast", "p_fast", key_type="direct")

    result = scanner.scan("accounts", top_n=2)

    assert [h.primary_key for h in result.hits] == ["K1", "K2"]
    assert list(result.sources_summary) == ["slow", "fast"]
    assert all(list(h.sources) == ["slow", "fast"] for h in result.hits)
