# Copyright (C) 2026 Karol Kedzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tests for generalized dimension blocks (g/t/s)."""
from __future__ import annotations

import numpy as np
import pytest


# ── Sub-task 2a: PatternConfig fields ──────────────────────────────


class TestPatternConfigDimensionBlockFields:
    """PatternConfig accepts the three new dimension block fields."""

    def test_pattern_config_has_dimension_block_fields(self):
        from hypertopos.cli.schema import PatternConfig

        pc = PatternConfig(
            type="anchor",
            entity_line="accounts",
            geo_properties=["lat", "lon"],
            metric_properties=["balance", "income"],
            semantic_dim={"columns": ["emb_0", "emb_1"], "n_components": 2},
        )
        assert pc.geo_properties == ["lat", "lon"]
        assert pc.metric_properties == ["balance", "income"]
        assert pc.semantic_dim == {
            "columns": ["emb_0", "emb_1"],
            "n_components": 2,
        }

    def test_pattern_config_blocks_default_none(self):
        from hypertopos.cli.schema import PatternConfig

        pc = PatternConfig(type="anchor", entity_line="accounts")
        assert pc.geo_properties is None
        assert pc.metric_properties is None
        assert pc.semantic_dim is None


# ── Sub-task 2b: Normalization functions ───────────────────────────


class TestNormalizeMetricBlock:
    """normalize_metric_block: z-score normalization."""

    def test_normalize_metric_block(self):
        from hypertopos.builder.dim_blocks import normalize_metric_block

        rng = np.random.default_rng(42)
        values = rng.normal(50.0, 10.0, size=(200, 3)).astype(np.float32)

        normalized, mu, sigma = normalize_metric_block(values)

        assert normalized.shape == values.shape
        # After z-scoring, column means should be near 0
        col_means = normalized.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=0.15)
        # Column stds should be near 1
        col_stds = normalized.std(axis=0)
        np.testing.assert_allclose(col_stds, 1.0, atol=0.15)

    def test_normalize_metric_block_handles_zero_variance(self):
        from hypertopos.builder.dim_blocks import normalize_metric_block

        values = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], dtype=np.float32)

        normalized, mu, sigma = normalize_metric_block(values)

        # First column is constant (5.0), sigma should be floored to 1.0
        assert sigma[0] == 1.0
        # Normalized constant col should be (5-5)/1 = 0
        np.testing.assert_allclose(normalized[:, 0], 0.0, atol=1e-6)
        # Second column has variance, should normalize normally
        assert sigma[1] > 0


class TestNormalizeGeoBlock:
    """normalize_geo_block: delegates to metric normalization."""

    def test_normalize_geo_block(self):
        from hypertopos.builder.dim_blocks import normalize_geo_block

        rng = np.random.default_rng(99)
        values = rng.uniform(-90, 90, size=(100, 2)).astype(np.float32)

        normalized, mu, sigma = normalize_geo_block(values)

        assert normalized.shape == values.shape
        col_means = normalized.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=0.2)


class TestNormalizeSemanticBlock:
    """normalize_semantic_block: PCA + z-score."""

    def test_normalize_semantic_block_pca(self):
        from hypertopos.builder.dim_blocks import normalize_semantic_block

        rng = np.random.default_rng(123)
        # 128-dim embeddings, reduce to 8
        values = rng.normal(0, 1, size=(500, 128)).astype(np.float32)

        normalized, mu, sigma, pca_components = normalize_semantic_block(
            values, n_components=8,
        )

        assert normalized.shape == (500, 8)
        assert pca_components.shape == (8, 128)
        # After PCA + z-score, means should be near 0
        col_means = normalized.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=0.15)

    def test_normalize_semantic_block_clamps_components(self):
        from hypertopos.builder.dim_blocks import normalize_semantic_block

        rng = np.random.default_rng(7)
        # 5-dim data, request 10 components -> should clamp to 5
        values = rng.normal(0, 1, size=(100, 5)).astype(np.float32)

        normalized, mu, sigma, pca_components = normalize_semantic_block(
            values, n_components=10,
        )

        assert normalized.shape == (100, 5)
        assert pca_components.shape == (5, 5)


# ── Sub-task 2c: Builder integration ──────────────────────────────


class TestBuilderDimBlockIntegration:
    """Integration: metric_properties appear in the shape vector."""

    def test_builder_includes_metric_dims_in_shape_vector(self, tmp_path):
        import pyarrow as pa

        from hypertopos.builder.builder import GDSBuilder, RelationSpec

        builder = GDSBuilder(
            sphere_id="test_dim_blocks",
            output_path=str(tmp_path / "gds_test"),
        )

        # Create a simple anchor line with numeric columns
        table = pa.table({
            "primary_key": [f"e{i}" for i in range(50)],
            "balance": np.random.default_rng(1).normal(1000, 200, 50).tolist(),
            "income": np.random.default_rng(2).normal(5000, 1000, 50).tolist(),
        })
        builder.add_line(
            "accounts", table, key_col="primary_key",
            source_id="s1", role="anchor",
        )

        # Create a minimal event line for a relation
        event_table = pa.table({
            "primary_key": [f"t{i}" for i in range(100)],
            "account_id": [f"e{i % 50}" for i in range(100)],
        })
        builder.add_line(
            "txns", event_table, key_col="primary_key",
            source_id="s2", role="event",
        )

        # Count FK on entity table for the relation
        from collections import Counter
        fk_counts = Counter(f"e{i % 50}" for i in range(100))
        count_col = [fk_counts.get(f"e{i}", 0) for i in range(50)]
        table = table.append_column(
            "_fk_txns_count",
            pa.array(count_col, type=pa.int64()),
        )
        builder._lines["accounts"].table = table

        builder.add_pattern(
            "p_accounts",
            pattern_type="anchor",
            entity_line="accounts",
            relations=[
                RelationSpec(
                    line_id="txns",
                    fk_col="_fk_txns_count",
                    direction="in",
                    required=False,
                    edge_max=10,
                ),
            ],
            metric_properties=["balance", "income"],
        )

        stats = builder._compute_population_stats(
            builder._patterns["p_accounts"],
        )

        # Shape = 1 relation + 2 metric dims = 3 total
        assert stats.mu.shape[0] == 3
        assert stats.sigma.shape[0] == 3

    def test_builder_includes_geo_dims_in_shape_vector(self, tmp_path):
        import pyarrow as pa

        from hypertopos.builder.builder import GDSBuilder, RelationSpec

        builder = GDSBuilder(
            sphere_id="test_geo_blocks",
            output_path=str(tmp_path / "gds_test_geo"),
        )

        rng = np.random.default_rng(42)
        table = pa.table({
            "primary_key": [f"e{i}" for i in range(50)],
            "lat": rng.uniform(40, 50, 50).tolist(),
            "lon": rng.uniform(-5, 5, 50).tolist(),
        })
        builder.add_line(
            "places", table, key_col="primary_key",
            source_id="s1", role="anchor",
        )

        builder.add_pattern(
            "p_places",
            pattern_type="anchor",
            entity_line="places",
            relations=[],
            geo_properties=["lat", "lon"],
        )

        stats = builder._compute_population_stats(
            builder._patterns["p_places"],
        )

        # Shape = 0 relations + 2 geo dims = 2 total
        assert stats.mu.shape[0] == 2
        assert stats.sigma.shape[0] == 2

    def test_builder_includes_semantic_dims_in_shape_vector(self, tmp_path):
        import pyarrow as pa

        from hypertopos.builder.builder import GDSBuilder, RelationSpec

        builder = GDSBuilder(
            sphere_id="test_sem_blocks",
            output_path=str(tmp_path / "gds_test_sem"),
        )

        rng = np.random.default_rng(7)
        emb = rng.normal(0, 1, size=(50, 16))
        cols = {
            "primary_key": [f"e{i}" for i in range(50)],
        }
        for d in range(16):
            cols[f"emb_{d}"] = emb[:, d].tolist()

        table = pa.table(cols)
        builder.add_line(
            "docs", table, key_col="primary_key",
            source_id="s1", role="anchor",
        )

        builder.add_pattern(
            "p_docs",
            pattern_type="anchor",
            entity_line="docs",
            relations=[],
            semantic_dim={
                "columns": [f"emb_{d}" for d in range(16)],
                "n_components": 4,
            },
        )

        stats = builder._compute_population_stats(
            builder._patterns["p_docs"],
        )

        # Shape = 0 relations + 4 PCA components = 4 total
        assert stats.mu.shape[0] == 4
        assert stats.sigma.shape[0] == 4

    def test_yaml_parses_dimension_block_keys(self, tmp_path):
        """YAML parsing: geo_properties, metric_properties, semantic_dim."""
        import yaml

        from hypertopos.cli.schema import parse_config

        cfg_path = tmp_path / "sphere.yaml"
        cfg_path.write_text(yaml.dump({
            "sphere_id": "test_blocks",
            "version": "0.1.0",
            "sources": {
                "s1": {"path": "dummy.csv"},
            },
            "lines": {
                "accounts": {
                    "source": "s1",
                    "key": "id",
                    "role": "anchor",
                },
            },
            "patterns": {
                "p_accounts": {
                    "type": "anchor",
                    "entity_line": "accounts",
                    "geo_properties": ["lat", "lon"],
                    "metric_properties": ["balance"],
                    "semantic_dim": {
                        "columns": ["emb_0", "emb_1"],
                        "n_components": 2,
                    },
                },
            },
        }), encoding="utf-8")

        cfg = parse_config(str(cfg_path))
        pat = cfg.patterns["p_accounts"]
        assert pat.geo_properties == ["lat", "lon"]
        assert pat.metric_properties == ["balance"]
        assert pat.semantic_dim == {
            "columns": ["emb_0", "emb_1"],
            "n_components": 2,
        }
