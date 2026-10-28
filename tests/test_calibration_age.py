from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from hypertopos.navigation.navigator import CALIBRATION_MAX_AGE_DAYS, GDSNavigator


def _navigator(last_calibrated_at: datetime | None) -> GDSNavigator:
    nav = GDSNavigator.__new__(GDSNavigator)
    pattern = SimpleNamespace(last_calibrated_at=last_calibrated_at)
    nav._storage = SimpleNamespace(  # type: ignore[assignment]
        read_calibration_tracker=lambda pid: None,
        read_sphere=lambda: SimpleNamespace(patterns={"p": pattern}),
    )
    return nav


def test_calibration_older_than_max_age_alerts():
    nav = _navigator(datetime.now(UTC) - timedelta(days=CALIBRATION_MAX_AGE_DAYS + 30))
    alerts = nav._check_calibration_staleness("p")
    assert [a["check_type"] for a in alerts] == ["calibration_age"]
    assert alerts[0]["details"]["age_days"] >= CALIBRATION_MAX_AGE_DAYS + 30
    assert alerts[0]["details"]["max_age_days"] == CALIBRATION_MAX_AGE_DAYS
    assert "recalibrate('p')" in alerts[0]["message"]


def test_recent_calibration_has_no_age_alert():
    assert (
        _navigator(datetime.now(UTC) - timedelta(days=10))._check_calibration_staleness("p") == []
    )


def test_unknown_calibration_time_has_no_age_alert():
    assert _navigator(None)._check_calibration_staleness("p") == []


def test_naive_timestamp_is_read_as_utc():
    stale = (datetime.now(UTC) - timedelta(days=CALIBRATION_MAX_AGE_DAYS + 1)).replace(tzinfo=None)
    assert len(_navigator(stale)._check_calibration_staleness("p")) == 1
