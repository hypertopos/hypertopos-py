# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Shared file-suffix detection for the data loaders (CLI + builder).

`Path.suffixes` returns *every* dotted segment, so a data file whose stem
contains a period — e.g. ``q1.2024.parquet`` — yields ``.2024.parquet`` and
fails to match any extension branch. The loaders only care about the real
trailing extension, except for the legitimate ``.gz``-compound form
(``data.csv.gz``). `normalized_suffix` returns the true extension while
preserving a ``<ext>.gz`` compound, so ``q1.2024.parquet`` → ``.parquet`` and
``q1.2024.csv.gz`` → ``.csv.gz``.
"""
from __future__ import annotations

from pathlib import Path


def normalized_suffix(path: str | Path) -> str:
    """Return the lower-cased trailing file extension, preserving ``<ext>.gz``.

    Examples::

        data.parquet      -> ".parquet"
        q1.2024.parquet   -> ".parquet"   (dotted stem ignored)
        data.csv.gz       -> ".csv.gz"
        q1.2024.csv.gz    -> ".csv.gz"    (dotted stem ignored, compound kept)
        archive           -> ""
    """
    suffixes = Path(path).suffixes
    if not suffixes:
        return ""
    last = suffixes[-1].lower()
    if last == ".gz" and len(suffixes) >= 2:
        return (suffixes[-2] + suffixes[-1]).lower()
    return last
