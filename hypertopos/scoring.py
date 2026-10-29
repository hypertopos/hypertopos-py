# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Population calibration held in memory: fit on an entity table, score any table.

``FrozenCalibration`` is the calibration a sphere build computes for one anchor
pattern whose dimensions are a metric block, without writing a sphere: the
calibrated evidence score (a frozen reference of the population on the
dimension values, its conformal p-value and the flag at a population
percentile) and the per-dimension mean and spread, optionally per group, that
the deviations explaining a score are measured with. With ``score_per_group``
each group also gets its own reference, so a row is scored within its group. It
runs the builder's own steps, so a sphere built from the same table reports the
same ``anomaly_score``, ``conformal_p``, ``score_rank_pct``, ``is_anomaly`` and
``delta_norm``. It is plain numpy and picklable: it can be broadcast to Spark executors.
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
from hypertopos.engine.robust import POOLED_GROUP as POOLED_GROUP  # re-exported
from hypertopos.engine.robust import (
    Reference,
    RobustScores,
    dim_shares_rows,
    fit_grouped_reference,
    fit_reference,
    is_anomaly,
    min_reference_size,
    pool_groups,
)
from hypertopos.engine.robust import score as score_reference

MIN_GROUP_SIZE = 100
"""Groups smaller than this are calibrated together as ``POOLED_GROUP``: a
group's mean and spread need enough members to measure a deviation against.
With ``score_per_group`` the default is the reference size instead
(:func:`hypertopos.engine.robust.min_reference_size`)."""
SHRINK_WEIGHT = 3.0
"""Pseudo-count of a shrunk dimension: a share or spread measured on ``n`` events
becomes ``(n * value + SHRINK_WEIGHT * population_mean) / (n + SHRINK_WEIGHT)``,
so one or two events cannot place an entity at an extreme by themselves."""


@dataclass(frozen=True)
class GroupCalibration:
    mu: np.ndarray  # standardised units
    sigma: np.ndarray
    size: int
    reference: Reference | None = None  # the group's own score; None: the population's


@dataclass(frozen=True)
class Scores:
    """Scores of the rows of one table, in table order."""

    anomaly_score: np.ndarray  # (n,) calibrated evidence score (mean of the axis ranks)
    conformal_p: np.ndarray  # (n,) conformal p of the score against the fitted population
    score_rank_pct: np.ndarray  # (n,) share of the population with a smaller score, 0-100
    is_anomaly: np.ndarray  # (n,) bool: conformal_p at the calibration's percentile
    delta: np.ndarray  # (n, D) z-scored deviations (group-relative with groups)
    delta_norm: np.ndarray  # (n,)
    score_delta: np.ndarray  # (n, D) float32 rank-Gaussian coordinates the score is computed on
    calibration_scope: np.ndarray | None  # (n,) "group" | "pooled" | "global"; None without groups


@dataclass(frozen=True)
class FrozenCalibration:
    dims: tuple[str, ...]
    percentile: float
    pre_mu: np.ndarray  # global standardisation of the dimensions, in their own units
    pre_sigma: np.ndarray
    mu: np.ndarray  # population statistics in standardised units
    sigma: np.ndarray
    size: int  # rows of the fitted population
    reference: Reference  # the calibrated evidence score of the fitted population
    group_by: str | None = None
    groups: dict[str, GroupCalibration] | None = None
    pooled: frozenset[str] = frozenset()  # groups calibrated inside POOLED_GROUP
    # dim -> (sample-size column, frozen population mean) for small-count shrinkage
    shrink: dict[str, tuple[str, float]] | None = None
    score_per_group: bool = False  # each group scored against its own reference
    min_group_size: int | None = None  # the pooling size, with score_per_group

    @classmethod
    def fit(
        cls,
        table: pa.Table,
        dims: Sequence[str],
        *,
        percentile: float = 99.0,
        group_by: str | None = None,
        min_group_size: int | None = None,
        shrink: Mapping[str, str] | None = None,
        score_per_group: bool = False,
    ) -> FrozenCalibration:
        """Calibrate on ``table``; nulls in a dimension count as 0, as in a build.

        The score's reference is fitted on the dimension values (after
        shrinkage) over the whole population, also with ``group_by``. There are
        no events, so the score has the ``mahal`` axis only, as in a build of
        the same table. The deviations go through the steps of a sphere's metric
        block: a global standardisation, then the population (or per-group)
        statistics. Scoring repeats the same arithmetic, so the fitted
        population scores to exactly the values a build stores.

        ``shrink`` maps a dimension to the column holding its sample size; the
        value is pulled toward the population mean with weight ``SHRINK_WEIGHT``
        (the mean is frozen here, so scoring new rows uses the same prior).

        ``score_per_group`` (needs ``group_by``): each group is scored against
        a reference fitted on its own rows, so its p-values, ranks and flags
        are within the group. Groups under ``min_group_size`` (default and
        lower bound: the reference size at ``percentile``) are pooled for both
        the score and the deviations; a pool still under it, and a group
        unseen at fit time, are scored against the population. Without it,
        groups under ``min_group_size`` (default ``MIN_GROUP_SIZE``) are pooled
        for the deviations only.
        """
        if score_per_group and group_by is None:
            raise ValueError("score_per_group needs group_by")
        group_size = effective_min_group_size(min_group_size, percentile, score_per_group)
        priors = {
            dim: (count, float(_column(table, dim).mean())) for dim, count in (shrink or {}).items()
        }
        shapes = _shapes(table, dims, priors)
        x = shapes.astype(np.float64)
        references: dict[str, Reference] = {}
        if score_per_group and group_by is not None:
            grouped, _, _ = fit_grouped_reference(
                x,
                tuple(dims),
                _group_ids(table, group_by),
                percentile=percentile,
                min_group_size=group_size,
            )
            reference, references = grouped.population, grouped.groups
        else:
            reference, _ = fit_reference(x, tuple(dims))
        standard, pre_mu, pre_sigma = normalize_metric_block(shapes)
        mu, sigma, _, _, _ = compute_stats(standard, percentile)
        groups: dict[str, GroupCalibration] | None = None
        pooled: frozenset[str] = frozenset()
        if group_by is not None:
            raw_ids = _group_ids(table, group_by)
            if score_per_group:
                ids, pooled = pool_groups(raw_ids, percentile, group_size)
            else:
                ids = pool_small_groups(raw_ids, group_size)
                pooled = frozenset(raw_ids[ids != raw_ids].tolist())
            stats, _, _ = compute_stats_grouped(standard, ids, percentile)
            groups = {
                gid: GroupCalibration(g_mu, g_sigma, size, references.get(gid))
                for gid, (g_mu, g_sigma, _, size) in stats.items()
            }
        return cls(
            dims=tuple(dims),
            percentile=percentile,
            pre_mu=pre_mu,
            pre_sigma=pre_sigma,
            mu=mu,
            sigma=sigma,
            size=table.num_rows,
            reference=reference,
            group_by=group_by,
            groups=groups,
            pooled=pooled,
            shrink=priors or None,
            score_per_group=score_per_group,
            min_group_size=group_size if score_per_group else None,
        )

    def score(self, table: pa.Table, *, members: bool = False) -> Scores:
        """Score ``table`` against the frozen population (no recalibration).

        New entities (the default) are calibrated against out-of-fold scores of
        the population, so they keep the alarm budget; ``members=True`` rescores
        the fitted table itself with its in-sample p-values, as a build stores.
        """
        shapes = self.shapes(table)
        assignments = self._assignments(table)
        evidence = self._evidence(shapes.astype(np.float64), assignments, members)
        standard = _deltas(shapes, self.pre_mu, self.pre_sigma)
        n = table.num_rows
        delta = np.empty_like(standard)
        scope: np.ndarray | None = (
            np.empty(n, dtype=object) if self.groups is not None and self.group_by else None
        )
        for mask, cal, label in assignments:
            delta[mask] = _deltas(standard[mask], cal.mu, cal.sigma)
            if scope is not None:
                scope[mask] = label
        norm = np.sqrt(np.einsum("ij,ij->i", delta, delta)).astype(np.float32)
        return Scores(
            anomaly_score=evidence.anomaly_score,
            conformal_p=evidence.conformal_p,
            score_rank_pct=evidence.score_rank_pct,
            is_anomaly=is_anomaly(evidence.conformal_p, self.percentile),
            delta=delta,
            delta_norm=norm,
            score_delta=evidence.delta,
            calibration_scope=scope,
        )

    def score_shares(self, table: pa.Table, score_delta: np.ndarray) -> np.ndarray:
        """:func:`score_shares` of each row of ``table`` against the reference
        that scored it (its group's with ``score_per_group``)."""
        if not self.score_per_group:
            return score_shares(self.reference, score_delta)
        out = np.zeros(np.shape(score_delta), dtype=np.float64)
        for mask, ref in self._references(self._assignments(table)):
            out[mask] = dim_shares_rows(ref, score_delta[mask])
        return out

    def _evidence(
        self,
        x: np.ndarray,
        assignments: list[tuple[np.ndarray, GroupCalibration, str]],
        members: bool,
    ) -> RobustScores:
        if not self.score_per_group or len(x) == 0:
            return score_reference(self.reference, x, members=members)
        # Each row's evidence from the reference that scores it: p, rank and flag
        # are within the row's group.
        parts = [
            (mask, score_reference(ref, x[mask], members=members))
            for mask, ref in self._references(assignments)
        ]
        merged = {}
        for name in (
            "delta",
            "delta_norm",
            "axis_ranks",
            "anomaly_score",
            "conformal_p",
            "score_rank_pct",
        ):
            first = getattr(parts[0][1], name)
            out = np.empty((len(x), *first.shape[1:]), dtype=first.dtype)
            for mask, part in parts:
                out[mask] = getattr(part, name)
            merged[name] = out
        return RobustScores(axis_values={}, **merged)

    def _references(
        self, assignments: list[tuple[np.ndarray, GroupCalibration, str]]
    ) -> list[tuple[np.ndarray, Reference]]:
        return [
            (mask, cal.reference if cal.reference is not None else self.reference)
            for mask, cal, _ in assignments
        ]

    def shapes(self, table: pa.Table) -> np.ndarray:
        """The calibrated dimensions of ``table`` as an (n, D) float32 matrix, shrunk."""
        return _shapes(table, self.dims, self.shrink or {})

    def row_params(self, table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Per-row standardised ``mu`` and ``sigma`` (n, D) and scope.

        Each row gets its group's calibration, the pool's for a group pooled at
        fit time, and the global one for a group never seen (scope ``global``).
        """
        n = table.num_rows
        d = len(self.dims)
        mu = np.empty((n, d), dtype=np.float32)
        sigma = np.empty((n, d), dtype=np.float32)
        scope: np.ndarray | None = (
            np.empty(n, dtype=object) if self.groups is not None and self.group_by else None
        )
        for mask, cal, label in self._assignments(table):
            mu[mask], sigma[mask] = cal.mu, cal.sigma
            if scope is not None:
                scope[mask] = label
        return mu, sigma, scope

    def _assignments(self, table: pa.Table) -> list[tuple[np.ndarray, GroupCalibration, str]]:
        """Row masks with the calibration each uses: its group, the pool, or global.

        With ``score_per_group`` the scope names where the score came from: a
        pool too small for its own reference is scored globally, while its
        deviations still use the pool's mean and spread.
        """
        everyone = GroupCalibration(self.mu, self.sigma, self.size)
        n = table.num_rows
        if self.groups is None or self.group_by is None:
            return [(np.ones(n, dtype=bool), everyone, "global")]
        raw = _group_ids(table, self.group_by)
        out: list[tuple[np.ndarray, GroupCalibration, str]] = []
        for gid in np.unique(raw):
            mask = raw == gid
            if gid in self.pooled:
                cal, label = self.groups[POOLED_GROUP], "pooled"
            elif gid in self.groups:
                cal, label = self.groups[gid], "group"
            else:
                cal, label = everyone, "global"  # unseen at fit time
            if self.score_per_group and cal.reference is None:
                label = "global"
            out.append((mask, cal, label))
        return out

    def raw_params(self, table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
        """Per-row ``mu`` and ``sigma`` in the dimensions' own units.

        Standardising ``x`` with these gives the row's deviations (``Scores.delta``).
        """
        mu, sigma, _ = self.row_params(table)
        return self.pre_mu + mu * self.pre_sigma, sigma * self.pre_sigma


def score_shares(reference: Reference, score_delta: np.ndarray) -> np.ndarray:
    """Each dimension's share of the score's ``mahal`` evidence, per row (n, D).

    The same Shapley split :func:`hypertopos.engine.robust.dim_shares` gives one
    entity (and ``explain_anomaly`` reports on a sphere), for many rows at once.
    A row at the centre of the score (zero distance) has no share.
    """
    return dim_shares_rows(reference, score_delta)


def effective_min_group_size(
    min_group_size: int | None, percentile: float, score_per_group: bool
) -> int:
    """The group size pooling uses: ``min_group_size``, or its default.

    The default is ``MIN_GROUP_SIZE``; with ``score_per_group`` it is the
    reference size at ``percentile``, and a smaller value raises, because a
    group under it could never flag a member.
    """
    if not score_per_group:
        return MIN_GROUP_SIZE if min_group_size is None else min_group_size
    bound = min_reference_size(percentile)
    if min_group_size is not None and min_group_size < bound:
        raise ValueError(
            f"min_group_size={min_group_size} is below {bound}, the smallest group "
            f"that can flag a member at the {percentile:g}th percentile"
        )
    return max(bound, min_group_size or 0)


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
    deltas: np.ndarray = calibrated_deltas(shapes, mu=mu, sigma_diag=sigma, dimension_weights=None)
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
