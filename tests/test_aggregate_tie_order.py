"""Equal aggregate values are ordered by group key, so the page cut is reproducible."""

from __future__ import annotations

import random

import pytest

from hypertopos.engine.aggregation import _rank_groups


@pytest.mark.parametrize("ascending", [True, False])
def test_ties_ordered_by_key_regardless_of_input_order(ascending: bool) -> None:
    items = [("b", 5.0), ("a", 5.0), ("c", 1.0), ("d", 9.0), ("e", 5.0)]
    expected = (
        [("c", 1.0), ("a", 5.0), ("b", 5.0), ("e", 5.0), ("d", 9.0)]
        if ascending
        else [("d", 9.0), ("a", 5.0), ("b", 5.0), ("e", 5.0), ("c", 1.0)]
    )
    for seed in range(5):
        shuffled = items[:]
        random.Random(seed).shuffle(shuffled)
        assert _rank_groups(shuffled, ascending) == expected


def test_tuple_group_keys_are_supported() -> None:
    items = [(("x", "2"), 3), (("x", "1"), 3)]
    assert _rank_groups(items, ascending=False) == [(("x", "1"), 3), (("x", "2"), 3)]
