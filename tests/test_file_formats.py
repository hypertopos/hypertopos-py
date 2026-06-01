# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Tests for normalized_suffix — the shared file-extension detector.

Pins the contract that fixes the dotted-stem bug (``q1.2024.parquet`` was
rejected because ``"".join(Path.suffixes)`` returned ``.2024.parquet``) while
preserving the legitimate ``.csv.gz`` compound.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypertopos._file_formats import normalized_suffix


@pytest.mark.parametrize(
    "name, expected",
    [
        # Simple extensions — must resolve exactly as before.
        ("data.parquet", ".parquet"),
        ("data.pq", ".pq"),
        ("data.csv", ".csv"),
        ("data.arrow", ".arrow"),
        ("data.arrows", ".arrows"),
        ("data.ipc", ".ipc"),
        ("data.feather", ".feather"),
        ("data.tsv", ".tsv"),
        # Compound .gz — must be preserved.
        ("data.csv.gz", ".csv.gz"),
        # The bug: dotted stem must collapse to the true extension.
        ("q1.2024.parquet", ".parquet"),
        ("2024.01.account_pairs.pq", ".pq"),
        # The nasty case: dotted stem AND compound together.
        ("q1.2024.csv.gz", ".csv.gz"),
        # Case-insensitivity.
        ("DATA.PARQUET", ".parquet"),
        ("Q1.2024.CSV.GZ", ".csv.gz"),
        # No suffix.
        ("archive", ""),
    ],
)
def test_normalized_suffix(name: str, expected: str) -> None:
    assert normalized_suffix(name) == expected
    assert normalized_suffix(Path(name)) == expected


@pytest.mark.parametrize(
    "name",
    [
        "data.parquet", "data.pq", "data.csv", "data.csv.gz", "data.arrow",
        "data.arrows", "data.ipc", "data.feather", "data.tsv",
    ],
)
def test_no_regression_vs_old_join_suffixes(name: str) -> None:
    """For every CURRENTLY-VALID suffix (no dotted stem), the new extractor
    returns the same value the old ``"".join(p.suffixes).lower()`` did — so the
    only behavior change is the dotted-stem wrong-reject becoming a correct
    accept. Blast-radius proof for this non-signature change."""
    old = "".join(Path(name).suffixes).lower()
    assert normalized_suffix(name) == old
