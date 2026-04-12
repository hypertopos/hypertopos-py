# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for adaptive chunk sizing and memory detection."""
from __future__ import annotations

import pytest

from hypertopos.builder.builder import GDSBuilder


class TestChunkSizing:
    def test_small_sphere_no_chunking(self):
        chunk = GDSBuilder._compute_chunk_size(
            n_entities=1000, n_buckets=50, n_dims=10,
            memory_budget_bytes=4 * 1024**3,
        )
        assert chunk == 1000

    def test_large_sphere_chunks(self):
        chunk = GDSBuilder._compute_chunk_size(
            n_entities=500_000, n_buckets=500, n_dims=28,
            memory_budget_bytes=4 * 1024**3,
        )
        assert 1000 <= chunk < 500_000

    def test_minimum_chunk_floor(self):
        chunk = GDSBuilder._compute_chunk_size(
            n_entities=500_000, n_buckets=500, n_dims=28,
            memory_budget_bytes=1 * 1024**2,
        )
        assert chunk == 1000

    def test_small_population_capped_at_n(self):
        # Population smaller than floor — returns n_entities (not floor)
        chunk = GDSBuilder._compute_chunk_size(
            n_entities=500, n_buckets=10, n_dims=5,
            memory_budget_bytes=100 * 1024**2,
        )
        assert chunk == 500

    def test_very_wide_dims(self):
        chunk = GDSBuilder._compute_chunk_size(
            n_entities=100_000, n_buckets=100, n_dims=200,
            memory_budget_bytes=2 * 1024**3,
        )
        assert 1000 <= chunk < 100_000


class TestPlanExecution:
    def test_small_all_parallel(self):
        n_workers, chunk_sizes = GDSBuilder._plan_execution(
            patterns_info={
                "p1": (1000, 50, 10),
                "p2": (500, 50, 8),
            },
            available_ram=4 * 1024**3,
        )
        assert n_workers == 2
        assert chunk_sizes["p1"] == 1000
        assert chunk_sizes["p2"] == 500

    def test_large_reduces_workers(self):
        n_workers, chunk_sizes = GDSBuilder._plan_execution(
            patterns_info={
                "p1": (500_000, 500, 28),
                "p2": (1_000_000, 500, 6),
            },
            available_ram=8 * 1024**3,
        )
        assert n_workers >= 1
        for pid, cs in chunk_sizes.items():
            assert cs >= 1000

    def test_single_pattern(self):
        n_workers, chunk_sizes = GDSBuilder._plan_execution(
            patterns_info={"p1": (100, 10, 5)},
            available_ram=1 * 1024**3,
        )
        assert n_workers == 1
        assert chunk_sizes["p1"] == 100


class TestDetectMemory:
    def test_returns_positive_int(self):
        mem = GDSBuilder._detect_available_memory()
        assert isinstance(mem, int)
        assert mem > 0

    def test_at_least_fallback(self):
        mem = GDSBuilder._detect_available_memory()
        assert mem >= 1 * 1024**3  # at least 1 GB on any dev machine
