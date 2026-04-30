"""YAML parsing + validation for edge_dimensions block."""
from __future__ import annotations

import pytest

from hypertopos.builder.mapping import (
    EdgeDimensionsConfig,
    parse_edge_dimensions,
)


def test_parse_bare_string_uses_defaults():
    cfg = parse_edge_dimensions(["pair_edge_count"], pattern_type="event")
    assert isinstance(cfg, EdgeDimensionsConfig)
    assert "pair_edge_count" in cfg.dims
    assert cfg.dims["pair_edge_count"] == {}


def test_parse_dict_with_overrides():
    cfg = parse_edge_dimensions(
        [{"position_in_chain": {"min_position": 7}}],
        pattern_type="event",
    )
    assert cfg.dims["position_in_chain"]["min_position"] == 7


def test_parse_mixed_bare_and_overrides():
    cfg = parse_edge_dimensions(
        [
            "pair_edge_count",
            {"position_in_chain": {"min_position": 5}},
        ],
        pattern_type="event",
    )
    assert "pair_edge_count" in cfg.dims
    assert cfg.dims["position_in_chain"]["min_position"] == 5


def test_parse_defaults_filled_in():
    cfg = parse_edge_dimensions(
        ["pair_amount_zscore"], pattern_type="event",
    )
    assert cfg.dims["pair_amount_zscore"]["cv_threshold"] == 0.05
    assert cfg.dims["pair_amount_zscore"]["min_count"] == 3


def test_reject_anchor_pattern():
    with pytest.raises(ValueError, match="event patterns"):
        parse_edge_dimensions(["pair_edge_count"], pattern_type="anchor")


def test_reject_min_position_below_3():
    with pytest.raises(ValueError, match="min_position must be"):
        parse_edge_dimensions(
            [{"position_in_chain": {"min_position": 2}}],
            pattern_type="event",
        )


def test_min_position_3_accepted():
    cfg = parse_edge_dimensions(
        [{"position_in_chain": {"min_position": 3}}],
        pattern_type="event",
    )
    assert cfg.dims["position_in_chain"]["min_position"] == 3


def test_reject_duplicate_dim():
    with pytest.raises(ValueError, match="declared twice"):
        parse_edge_dimensions(
            ["pair_edge_count", "pair_edge_count"],
            pattern_type="event",
        )


def test_reject_unknown_dim():
    with pytest.raises(ValueError, match="unknown edge dimension"):
        parse_edge_dimensions(["nonsense_dim"], pattern_type="event")


def test_reject_cv_threshold_zero():
    with pytest.raises(ValueError, match="cv_threshold"):
        parse_edge_dimensions(
            [{"pair_amount_zscore": {"cv_threshold": 0.0, "min_count": 3}}],
            pattern_type="event",
        )


def test_reject_cv_threshold_above_1():
    with pytest.raises(ValueError, match="cv_threshold"):
        parse_edge_dimensions(
            [{"pair_amount_zscore": {"cv_threshold": 1.5, "min_count": 3}}],
            pattern_type="event",
        )


def test_reject_min_count_below_2():
    with pytest.raises(ValueError, match="min_count"):
        parse_edge_dimensions(
            [{"pair_amount_zscore": {"cv_threshold": 0.05, "min_count": 1}}],
            pattern_type="event",
        )


def test_reject_amt1_min_negative():
    with pytest.raises(ValueError, match="amt1_min"):
        parse_edge_dimensions(
            [{"find_motif_structuring": {
                "time_window_hours": 1.0,
                "amt1_min": -100.0, "amt2_max": 10000.0,
            }}],
            pattern_type="event",
        )


def test_reject_amt2_max_zero():
    with pytest.raises(ValueError, match="amt2_max"):
        parse_edge_dimensions(
            [{"find_motif_structuring": {
                "time_window_hours": 1.0,
                "amt1_min": 10000.0, "amt2_max": 0.0,
            }}],
            pattern_type="event",
        )


def test_reject_time_window_hours_zero():
    with pytest.raises(ValueError, match="time_window_hours"):
        parse_edge_dimensions(
            [{"find_motif_structuring": {
                "time_window_hours": 0.0,
                "amt1_min": 10000.0, "amt2_max": 10000.0,
            }}],
            pattern_type="event",
        )


def test_reject_burst_seconds_negative():
    with pytest.raises(ValueError, match="burst_seconds"):
        parse_edge_dimensions(
            [{"time_since_pair_last_edge": {
                "burst_seconds": -1.0, "dormant_seconds": 999.0,
            }}],
            pattern_type="event",
        )


def test_reject_non_list_input():
    with pytest.raises(ValueError, match="must be a list"):
        parse_edge_dimensions("pair_edge_count", pattern_type="event")  # type: ignore[arg-type]


def test_reject_malformed_entry():
    with pytest.raises(ValueError, match="must be a string or single-key dict"):
        parse_edge_dimensions(
            [{"a": {}, "b": {}}],   # dict with two keys is malformed
            pattern_type="event",
        )


def test_empty_list_returns_empty_config():
    cfg = parse_edge_dimensions([], pattern_type="event")
    assert cfg.dims == {}
