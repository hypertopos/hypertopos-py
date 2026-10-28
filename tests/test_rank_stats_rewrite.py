# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Population rank columns are rewritten in place, fragment by fragment.

``score_rank_pct`` and ``conformal_p`` are whole-population statistics, so
every append rewrites them for every row. A key-joined upsert of the whole
population builds one hash table over all rows and exhausts the query memory
pool on multi-million-row patterns; it also drops the key-lookup index. The
rewrite joins on row ids one fragment at a time and commits once.
"""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

from hypertopos.builder._stats import compute_conformal_p, population_left_ranks
from hypertopos.storage.writer import GDSWriter

PATTERN = "p"


def _geometry(tmp_path: Path) -> tuple[str, Path]:
    """Three fragments, one deletion vector, tied norms and a NaN norm."""
    base = tmp_path / "sphere"
    path = base / "geometry" / PATTERN / "data.lance"
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(7)
    norms = rng.gamma(2.0, 1.0, 900).astype(np.float32)
    norms[10:40] = 1.5  # ties across fragment boundaries below
    norms[310:330] = 1.5
    norms[77] = np.nan
    for i, lo in enumerate(range(0, 900, 300)):
        sl = slice(lo, lo + 300)
        table = pa.table(
            {
                "primary_key": pa.array([f"K{j:04d}" for j in range(lo, lo + 300)]),
                "delta_norm": pa.array(norms[sl], pa.float32()),
                "anomaly_score": pa.array(norms[sl].astype(np.float64), pa.float64()),
                "score_rank_pct": pa.array(np.full(300, -1.0), pa.float32()),
                "conformal_p": pa.array(np.full(300, -1.0), pa.float32()),
            }
        )
        lance.write_dataset(table, str(path), mode="create" if i == 0 else "append")
    ds = lance.dataset(str(path))
    ds.delete("primary_key IN ('K0005', 'K0450', 'K0899')")
    ds.create_scalar_index("primary_key", "BTREE")
    return str(base), path


def _expected(path: Path) -> dict[str, tuple[float, float]]:
    t = lance.dataset(str(path)).to_table(columns=["primary_key", "delta_norm"])
    norms = t["delta_norm"].to_numpy(zero_copy_only=False)
    n = len(norms)
    left = np.searchsorted(np.sort(norms), norms, side="left")
    rank = (left / n * 100).astype(np.float32)
    cp = ((n - left) / n).astype(np.float32)
    keys = t["primary_key"].to_pylist()
    return {k: (float(r), float(c)) for k, r, c in zip(keys, rank, cp, strict=True)}


def _stored(path: Path) -> dict[str, tuple[float, float]]:
    t = lance.dataset(str(path)).to_table(columns=["primary_key", "score_rank_pct", "conformal_p"])
    return {
        k: (float(r), float(c))
        for k, r, c in zip(
            t["primary_key"].to_pylist(),
            t["score_rank_pct"].to_pylist(),
            t["conformal_p"].to_pylist(),
            strict=True,
        )
    }


@pytest.mark.parametrize(
    "values",
    [
        np.array([], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
        np.array([2.0, 1.0, 2.0, np.nan, 0.5, 2.0, np.nan, 1.0], dtype=np.float32),
        np.random.default_rng(3).gamma(2.0, 1.0, 10_000).astype(np.float32).round(1),
    ],
    ids=["empty", "single", "ties_and_nan", "many_ties"],
)
def test_population_left_ranks_equals_searchsorted(values: np.ndarray) -> None:
    expected = np.searchsorted(np.sort(values), values, side="left")
    np.testing.assert_array_equal(population_left_ranks(values), expected)
    n = len(values)
    if n:
        np.testing.assert_array_equal(
            compute_conformal_p(values), ((n - expected) / n).astype(np.float32)
        )


def test_fixture_spans_several_fragments(tmp_path: Path) -> None:
    _, path = _geometry(tmp_path)
    ds = lance.dataset(str(path))
    assert len(ds.get_fragments()) == 3
    assert ds.count_rows() == 897


@pytest.mark.parametrize(
    ("method", "rank", "cp"),
    [
        ("recompute_rank_stats", True, True),
        ("recompute_score_rank_pct", True, False),
        ("recompute_conformal_p", False, True),
    ],
)
def test_rewrite_matches_population_formula(
    tmp_path: Path, method: str, rank: bool, cp: bool
) -> None:
    base, path = _geometry(tmp_path)
    getattr(GDSWriter(base), method)(PATTERN)

    expected, stored = _expected(path), _stored(path)
    assert stored.keys() == expected.keys()
    for key, (exp_rank, exp_cp) in expected.items():
        got_rank, got_cp = stored[key]
        if rank:
            assert got_rank == exp_rank, key
        else:
            assert got_rank == -1.0, key
        if cp:
            assert got_cp == exp_cp, key
        else:
            assert got_cp == -1.0, key


def test_rewrite_keeps_key_index_and_commits_once(tmp_path: Path) -> None:
    base, path = _geometry(tmp_path)
    before = lance.dataset(str(path))

    GDSWriter(base).recompute_rank_stats(PATTERN)

    after = lance.dataset(str(path))
    assert after.version == before.version + 1
    assert after.count_rows() == before.count_rows()
    pk_index = next(i for i in after.describe_indices() if i.name == "primary_key_idx")
    assert pk_index.num_rows_indexed == after.count_rows()


def test_rewrite_refreshes_index_on_rewritten_column(tmp_path: Path) -> None:
    base, path = _geometry(tmp_path)
    lance.dataset(str(path)).create_scalar_index("score_rank_pct", "BTREE")

    GDSWriter(base).recompute_rank_stats(PATTERN)

    ds = lance.dataset(str(path))
    rank_index = next(i for i in ds.describe_indices() if "score_rank_pct" in i.field_names)
    assert rank_index.num_rows_indexed == ds.count_rows()
    expected = sum(1 for r, _ in _expected(path).values() if r >= 90.0)
    assert ds.count_rows(filter="score_rank_pct >= 90.0") == expected
    # the -1.0 placeholders written by the fixture must not survive in the index
    assert ds.count_rows(filter="score_rank_pct < 0") == 0
