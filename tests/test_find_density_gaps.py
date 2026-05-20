"""Navigator-level integration for find_density_gaps.

Uses the bundled Berka sphere because synthetic GDSBuilder spheres
trigger Lance internal panics on small entity counts (the encoder
divides by zero on undersized rowgroups).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypertopos import HyperSphere


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BERKA_PATH = PROJECT_ROOT / "benchmark" / "berka" / "sphere" / "gds_berka_banking"


def _has_berka_3_0() -> bool:
    sphere_json = BERKA_PATH / "_gds_meta" / "sphere.json"
    if not sphere_json.exists():
        return False
    import json
    return json.loads(sphere_json.read_text()).get("format_version") == "3.0"


pytestmark = pytest.mark.skipif(
    not _has_berka_3_0(),
    reason="Berka sphere not built at format 3.0 (run benchmark/berka/sphere/sphere.yaml first)",
)


@pytest.fixture(scope="module")
def berka_nav():
    hs = HyperSphere.open(BERKA_PATH)
    return hs.session("dg-test").navigator()


def test_find_density_gaps_returns_dict_with_gaps(berka_nav):
    out = berka_nav.find_density_gaps(
        "account_behavior_pattern", top_n=10,
    )
    assert isinstance(out, dict)
    assert "gaps" in out
    assert "excluded_dims" in out
    assert "n_pairs_tested" in out
    assert out["n_entities"] >= 100


def test_find_density_gaps_rejects_unknown_pattern(berka_nav):
    with pytest.raises(Exception, match="pattern not found"):
        berka_nav.find_density_gaps("nonexistent")


def test_find_density_gaps_rejects_invalid_alpha(berka_nav):
    with pytest.raises(Exception, match="alpha"):
        berka_nav.find_density_gaps(
            "account_behavior_pattern", alpha=0.0,
        )


def test_find_density_gaps_rejects_invalid_bins(berka_nav):
    with pytest.raises(Exception, match="bins"):
        berka_nav.find_density_gaps(
            "account_behavior_pattern", bins=2,
        )


def test_find_density_gaps_rejects_invalid_r_window(berka_nav):
    with pytest.raises(Exception, match="r_min"):
        berka_nav.find_density_gaps(
            "account_behavior_pattern", r_min=0.5, r_max=0.4,
        )


def test_find_density_gaps_rejects_invalid_top_n(berka_nav):
    with pytest.raises(Exception, match="top_n"):
        berka_nav.find_density_gaps(
            "account_behavior_pattern", top_n=0,
        )


def test_find_density_gaps_returns_sorted_by_ratio(berka_nav):
    out = berka_nav.find_density_gaps(
        "account_behavior_pattern", top_n=20,
    )
    ratios = [g["ratio"] for g in out["gaps"]]
    assert ratios == sorted(ratios, reverse=True)


def test_find_density_gaps_unknown_dim_in_user_pairs(berka_nav):
    with pytest.raises(Exception, match="unknown dim names"):
        berka_nav.find_density_gaps(
            "account_behavior_pattern",
            dim_pairs=[("nope_x", "nope_y")],
        )


def test_find_density_gaps_includes_raw_ranges(berka_nav):
    out = berka_nav.find_density_gaps(
        "account_behavior_pattern", top_n=5,
    )
    for gap in out["gaps"]:
        assert "delta_range_i" in gap
        assert "delta_range_j" in gap
        assert len(gap["delta_range_i"]) == 2
        assert len(gap["delta_range_j"]) == 2
        assert gap["dim_i"] != gap["dim_j"]
