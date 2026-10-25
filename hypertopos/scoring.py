# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Population calibration held in memory: fit on an entity table, score any table.

``FrozenCalibration`` is the calibration a sphere build computes for one anchor
pattern whose dimensions are a metric block (per-dimension mean and spread, the
anomaly radius at a population percentile, optionally per group), without
writing a sphere. It runs the builder's own steps, so a sphere built from the
same table reports the same ``delta_norm``, ``is_anomaly``, rank and conformal
p-value. It is plain numpy and picklable: it can be broadcast to Spark executors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from hypertopos.builder._stats import compute_stats, compute_stats_grouped
from hypertopos.builder.dim_blocks import normalize_metric_block
from hypertopos.engine.geometry import calibrated_deltas

MIN_GROUP_SIZE = 100
"""Groups smaller than this are calibrated together as ``POOLED_GROUP``: a
percentile threshold needs about 1 / (1 - percentile / 100) members to flag
anyone, and a one-member group would never be flagged."""
POOLED_GROUP = "(small groups)"
SHRINK_WEIGHT = 3.0
"""Pseudo-count of a shrunk dimension: a share or spread measured on ``n`` events
becomes ``(n * value + SHRINK_WEIGHT * population_mean) / (n + SHRINK_WEIGHT)``,
so one or two events cannot place an entity at an extreme by themselves."""


@dataclass(frozen=True)
class GroupCalibration:
    mu: np.ndarray  # standardised units
    sigma: np.ndarray
    theta_norm: float
    size: int


@dataclass(frozen=True)
class Scores:
    """Scores of the rows of one table, in table order."""

    delta: np.ndarray  # (n, D) z-scored deviations
    delta_norm: np.ndarray  # (n,)
    is_anomaly: np.ndarray  # (n,) bool
    rank_pct: np.ndarray  # (n,) share of the population with a smaller delta_norm, 0-100
    conformal_p: np.ndarray  # (n,) share of the population with delta_norm >= this one
    calibration_scope: np.ndarray | None  # (n,) "group" | "pooled" | "global"; None without groups


@dataclass(frozen=True)
class FrozenCalibration:
    dims: tuple[str, ...]
    percentile: float
    pre_mu: np.ndarray  # global standardisation of the dimensions, in their own units
    pre_sigma: np.ndarray
    mu: np.ndarray  # population statistics in standardised units
    sigma: np.ndarray
    theta_norm: float
    population_norms: np.ndarray  # sorted delta_norm of the fitted population
    group_by: str | None = None
    groups: dict[str, GroupCalibration] | None = None
    pooled: frozenset[str] = frozenset()  # groups calibrated inside POOLED_GROUP
    # dim -> (sample-size column, frozen population mean) for small-count shrinkage
    shrink: dict[str, tuple[str, float]] | None = None

    @classmethod
    def fit(
        cls,
        table: pa.Table,
        dims: Sequence[str],
        *,
        percentile: float = 99.0,
        group_by: str | None = None,
        min_group_size: int = MIN_GROUP_SIZE,
        shrink: Mapping[str, str] | None = None,
    ) -> FrozenCalibration:
        """Calibrate on ``table``; nulls in a dimension count as 0, as in a build.

        The dimensions go through the steps of a sphere's metric block: a global
        standardisation, then the population (or per-group) statistics. Scoring
        repeats the same arithmetic, so the fitted population scores to exactly
        the norms it was ranked by.

        ``shrink`` maps a dimension to the column holding its sample size; the
        value is pulled toward the population mean with weight ``SHRINK_WEIGHT``
        (the mean is frozen here, so scoring new rows uses the same prior).
        """
        priors = {
            dim: (count, float(_column(table, dim).mean())) for dim, count in (shrink or {}).items()
        }
        shapes = _shapes(table, dims, priors)
        standard, pre_mu, pre_sigma = normalize_metric_block(shapes)
        mu, sigma, theta, _, norms, _ = compute_stats(standard, percentile)
        theta_norm = float(np.linalg.norm(theta))
        groups: dict[str, GroupCalibration] | None = None
        pooled: frozenset[str] = frozenset()
        if group_by is not None:
            raw_ids = _group_ids(table, group_by)
            ids = pool_small_groups(raw_ids, min_group_size)
            pooled = frozenset(raw_ids[ids != raw_ids].tolist())
            stats, _, norms = compute_stats_grouped(standard, ids, percentile)
            groups = {
                gid: GroupCalibration(g_mu, g_sigma, float(np.linalg.norm(g_theta)), size)
                for gid, (g_mu, g_sigma, g_theta, size) in stats.items()
            }
        return cls(
            dims=tuple(dims),
            percentile=percentile,
            pre_mu=pre_mu,
            pre_sigma=pre_sigma,
            mu=mu,
            sigma=sigma,
            theta_norm=theta_norm,
            population_norms=np.sort(norms.astype(np.float32)),
            group_by=group_by,
            groups=groups,
            pooled=pooled,
            shrink=priors or None,
        )

    def score(self, table: pa.Table) -> Scores:
        """Score ``table`` against the frozen population (no recalibration)."""
        standard = _deltas(self.shapes(table), self.pre_mu, self.pre_sigma)
        n = table.num_rows
        delta = np.empty_like(standard)
        theta_norm = np.empty(n, dtype=np.float64)
        scope: np.ndarray | None = (
            np.empty(n, dtype=object) if self.groups is not None and self.group_by else None
        )
        for mask, cal, label in self._assignments(table):
            delta[mask] = _deltas(standard[mask], cal.mu, cal.sigma)
            theta_norm[mask] = cal.theta_norm
            if scope is not None:
                scope[mask] = label
        norm = np.sqrt(np.einsum("ij,ij->i", delta, delta)).astype(np.float32)
        is_anomaly = (theta_norm > 0.0) & (norm >= theta_norm)
        population = len(self.population_norms)
        left = np.searchsorted(self.population_norms, norm, side="left")
        return Scores(
            delta=delta,
            delta_norm=norm,
            is_anomaly=is_anomaly,
            rank_pct=(left / population * 100).astype(np.float32),
            conformal_p=((population - left) / population).astype(np.float32),
            calibration_scope=scope,
        )

    def shapes(self, table: pa.Table) -> np.ndarray:
        """The calibrated dimensions of ``table`` as an (n, D) float32 matrix, shrunk."""
        return _shapes(table, self.dims, self.shrink or {})

    def row_params(
        self, table: pa.Table
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Per-row standardised ``mu`` and ``sigma`` (n, D), threshold norm and scope.

        Each row gets its group's calibration, the pool's for a group pooled at
        fit time, and the global one for a group never seen (scope ``global``).
        """
        n = table.num_rows
        d = len(self.dims)
        mu = np.empty((n, d), dtype=np.float32)
        sigma = np.empty((n, d), dtype=np.float32)
        theta_norm = np.empty(n, dtype=np.float64)
        scope: np.ndarray | None = (
            np.empty(n, dtype=object) if self.groups is not None and self.group_by else None
        )
        for mask, cal, label in self._assignments(table):
            mu[mask], sigma[mask], theta_norm[mask] = cal.mu, cal.sigma, cal.theta_norm
            if scope is not None:
                scope[mask] = label
        return mu, sigma, theta_norm, scope

    def _assignments(self, table: pa.Table) -> list[tuple[np.ndarray, GroupCalibration, str]]:
        """Row masks with the calibration each uses: its group, the pool, or global."""
        everyone = GroupCalibration(
            self.mu, self.sigma, self.theta_norm, len(self.population_norms)
        )
        n = table.num_rows
        if self.groups is None or self.group_by is None:
            return [(np.ones(n, dtype=bool), everyone, "global")]
        raw = _group_ids(table, self.group_by)
        out: list[tuple[np.ndarray, GroupCalibration, str]] = []
        for gid in np.unique(raw):
            mask = raw == gid
            if gid in self.pooled:
                out.append((mask, self.groups[POOLED_GROUP], "pooled"))
            elif gid in self.groups:
                out.append((mask, self.groups[gid], "group"))
            else:
                out.append((mask, everyone, "global"))  # unseen at fit time
        return out

    def raw_params(self, table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
        """Per-row ``mu`` and ``sigma`` in the dimensions' own units.

        Standardising ``x`` with these gives the row's deviations (``Scores.delta``).
        """
        mu, sigma, _, _ = self.row_params(table)
        return self.pre_mu + mu * self.pre_sigma, sigma * self.pre_sigma


def pool_small_groups(ids: np.ndarray, min_group_size: int) -> np.ndarray:
    """Relabel members of groups smaller than ``min_group_size`` to ``POOLED_GROUP``."""
    values, counts = np.unique(ids, return_counts=True)
    small = set(values[counts < min_group_size].tolist())
    if not small:
        return ids
    # object dtype: a fixed-width string array would truncate the longer label.
    pooled = ids.astype(object)
    pooled[np.isin(ids, list(small))] = POOLED_GROUP
    return pooled


def _deltas(shapes: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # The library's diagonal metric (sigma already floored by the statistics step).
    deltas: np.ndarray = calibrated_deltas(
        shapes, mu=mu, sigma_diag=sigma, cholesky_inv=None, dimension_weights=None
    )
    return deltas.astype(np.float32)


def _column(table: pa.Table, name: str) -> np.ndarray:
    values: np.ndarray = pc.fill_null(table[name], 0).to_numpy(zero_copy_only=False)
    return values.astype(np.float64)


def _shapes(
    table: pa.Table, dims: Sequence[str], shrink: Mapping[str, tuple[str, float]]
) -> np.ndarray:
    columns = []
    for d in dims:
        values = _column(table, d)
        if d in shrink:
            count_column, prior = shrink[d]
            n = _column(table, count_column)
            values = (n * values + SHRINK_WEIGHT * prior) / (n + SHRINK_WEIGHT)
        columns.append(values.astype(np.float32))
    return np.column_stack(columns) if columns else np.zeros((table.num_rows, 0), np.float32)


def _group_ids(table: pa.Table, group_by: str) -> np.ndarray:
    ids: np.ndarray = pc.cast(table[group_by], pa.string()).to_numpy(zero_copy_only=False)
    return ids.astype(str)
