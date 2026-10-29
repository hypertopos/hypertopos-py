# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Calibrated evidence score: a frozen population reference and scoring against it.

The score of an entity is the mean of its population ranks on a few axes:

- ``mahal``: Ledoit-Wolf Mahalanobis distance on rank-Gaussian coordinates of the
  pattern's dimensions (per-dimension CDF kept as a quantile sketch);
- ``current``: residual of the entity's displacement between the two calendar
  halves of the event period against the population's velocity field (mean
  displacement of the 50 nearest earlier states, the entity itself left out);
- ``knn`` (opt-in): mean distance to the 100 nearest reference members on 12 PCA
  dimensions of standardised sign-log1p coordinates.

The p-value is a conformal p-value against the fitted population's scores. Everything
the score needs is held in :class:`Reference`, so rows unseen at fit time (incremental
ingest, Spark) are scored without refitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtri

AXES_DEFAULT: tuple[str, ...] = ("mahal", "current")
AXES_ALL: tuple[str, ...] = ("mahal", "current", "knn")
CDF_POINTS = 4097
VELOCITY_K = 50
KNN_K = 100
KNN_DIMS = 12
MAX_REFERENCE_ROWS = 100_000
OOF_FOLDS = 5
# Above this many rows per dimension the in-sample calibration of new rows is
# already within noise of the budget, and the folds would cost a full refit each.
OOF_MAX_ROWS_PER_DIM = 2_000
_DAY = 86400.0


@dataclass(frozen=True)
class EventArrays:
    """Event rows of the scored entities: ``code`` is the entity's row index."""

    code: np.ndarray  # (m,) int64
    direction: np.ndarray  # (m,) int8, 0 = out, 1 = in
    ts: np.ndarray  # (m,) float64 epoch seconds
    amount: np.ndarray  # (m,) float64, raw


@dataclass(frozen=True)
class Reference:
    dims: tuple[str, ...]
    axes: tuple[str, ...]
    cdf_grid: np.ndarray  # (D, Q)
    cdf_levels: np.ndarray  # (Q,)
    lw_location: np.ndarray  # (D,)
    lw_precision: np.ndarray  # (D, D)
    axis_sorted: dict[str, np.ndarray]  # sorted finite axis values of the fitted population
    cal_sorted: np.ndarray  # sorted fused scores of the fitted population
    n_fitted: int
    t0: float | None = None
    t_mid: float | None = None
    vel_s0: np.ndarray | None = None  # (m, 5)
    vel_u: np.ndarray | None = None  # (m, 5)
    log_mu: np.ndarray | None = None  # (D,)
    log_sd: np.ndarray | None = None  # (D,)
    pca_mean: np.ndarray | None = None  # (D,)
    pca_components: np.ndarray | None = None  # (k, D)
    knn_ref: np.ndarray | None = None  # (r, k)
    # sorted fused scores of the fitted population over the axes other than
    # ``current``: the calibration of rows scored without events (e.g. ingest)
    cal_sorted_no_current: np.ndarray | None = None
    # Keys of rows present at fit time but left out of it (a recalibration with
    # excluded keys): they were scored as new rows.
    excluded_keys: np.ndarray | None = None
    # Out-of-fold fused scores of the population: what rows unseen at fit time
    # are calibrated against (members keep ``cal_sorted``).
    mahal_sorted_oof: np.ndarray | None = None
    cal_sorted_oof: np.ndarray | None = None
    cal_sorted_no_current_oof: np.ndarray | None = None


@dataclass(frozen=True)
class RobustScores:
    delta: np.ndarray  # (n, D) float32, rank-Gaussian deviation per named dimension
    delta_norm: np.ndarray  # (n,) float32, the Mahalanobis axis value
    axis_values: dict[str, np.ndarray]
    axis_ranks: np.ndarray  # (n, A) float32
    anomaly_score: np.ndarray  # (n,) float64
    conformal_p: np.ndarray  # (n,) float64
    score_rank_pct: np.ndarray  # (n,) float64, 0-100
    sample_index: np.ndarray | None = None  # fit only: rows drawn into the reference samples


def conformal_p(reference_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Conformal p-value against the fitted population: ``(1 + #{reference >= s}) / (n + 1)``."""
    n = len(reference_sorted)
    above = n - np.searchsorted(reference_sorted, scores, side="left")
    return (1.0 + above) / (n + 1.0)


def is_anomaly(p: np.ndarray, anomaly_percentile: float) -> np.ndarray:
    return np.asarray(p) <= 1.0 - anomaly_percentile / 100.0


def _levels(points: int) -> np.ndarray:
    return (np.arange(points) + 0.5) / points


def _quantiles_of_sorted(v: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """``np.quantile(v, levels)`` (linear) of an already sorted ``v``."""
    pos = levels * (len(v) - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, len(v) - 1)
    frac = pos - lo
    out: np.ndarray = v[lo] * (1.0 - frac) + v[hi] * frac
    return out


def _levels_of_sorted(g: np.ndarray, levels: np.ndarray, v: np.ndarray) -> np.ndarray:
    """CDF level of each value of the sorted ``v`` under the grid ``g``; ties take the mid level.

    The grid positions inside ``v`` give, by a cumulative count, how many grid
    points lie below / at each value in O(n + Q) instead of a search per value.
    """
    n = len(v)
    below = np.cumsum(np.bincount(np.searchsorted(v, g, side="right"), minlength=n + 1))[:n]
    at_or_below = np.cumsum(np.bincount(np.searchsorted(v, g, side="left"), minlength=n + 1))[:n]
    col: np.ndarray = np.interp(v, g, levels)
    tie = below < at_or_below
    col[tie] = (levels[below[tie]] + levels[at_or_below[tie] - 1]) / 2.0
    return col


def _sketch_and_gauss(
    x: np.ndarray, grid: np.ndarray | None, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One sort per column: the grid (fitted here when ``grid`` is None) and the coordinates."""
    xt = np.ascontiguousarray(np.asarray(x, dtype=np.float64).T)
    out_grid = np.empty((xt.shape[0], len(levels))) if grid is None else grid
    u = np.empty_like(xt)
    for d in range(xt.shape[0]):
        order = np.argsort(xt[d], kind="stable")
        v = xt[d][order]
        if grid is None:
            out_grid[d] = _quantiles_of_sorted(v, levels)
        u[d, order] = _levels_of_sorted(out_grid[d], levels, v)
    z = np.asarray(ndtri(np.clip(u, levels[0], levels[-1])), dtype=np.float64).T
    return out_grid, np.ascontiguousarray(z)


def fit_cdf(x: np.ndarray, points: int = CDF_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension quantile sketch: ``(grid (D, Q), levels (Q,))``."""
    levels = _levels(points)
    xt = np.asarray(x, dtype=np.float64).T
    grid = np.stack([_quantiles_of_sorted(np.sort(col), levels) for col in xt])
    return grid, levels


def to_gauss(grid: np.ndarray, levels: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank-Gaussian coordinates of ``x`` under the sketch; ties on the grid take the mid level."""
    return _sketch_and_gauss(x, grid, levels)[1]


def ledoit_wolf(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ledoit-Wolf shrunk covariance of ``z`` (n, D): ``(location, precision)``.

    Same estimator as scikit-learn's ``LedoitWolf`` (shrinkage toward the scaled
    identity, Ledoit and Wolf 2004), without the optional dependency.
    """
    location = z.mean(axis=0)
    x = z - location
    n, p = x.shape
    emp = x.T @ x / n
    x2 = x**2
    trace = x2.sum() / n
    mu = trace / p
    beta_ = float(np.sum(x2.T @ x2))
    delta_ = float(np.sum(emp**2))
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * trace + p * mu**2) / p
    shrinkage = 0.0 if beta == 0 or delta == 0 else min(beta, delta) / delta
    cov = (1.0 - shrinkage) * emp + shrinkage * mu * np.eye(p)
    return location, np.linalg.pinv(cov, hermitian=True)


def _principal_axes(e: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and top-``k`` principal directions (k, D) of ``e``."""
    mean = e.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(e - mean, rowvar=False).reshape(e.shape[1], e.shape[1]))
    order = np.argsort(vals)[::-1][:k]
    return mean, np.ascontiguousarray(vecs[:, order].T)


def event_states(
    events: EventArrays, n: int, t_mid: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """States per calendar half and the displacement: ``(s0 (n, 5), u (n, 5), active (n,))``.

    State columns: log1p count out, log1p sum out, log1p count in, log1p sum in,
    log1p count all; sums are of ``log1p(amount)``.
    """
    second = events.ts >= t_mid
    la = np.log1p(np.maximum(events.amount, 0.0))
    states = []
    counts = []
    for half in (False, True):
        sel = second == half
        cols = []
        for d in (0, 1):
            s = sel & (events.direction == d)
            cols.append(np.log1p(np.bincount(events.code[s], minlength=n)))
            cols.append(np.log1p(np.bincount(events.code[s], la[s], n)))
        c = np.bincount(events.code[sel], minlength=n)
        cols.append(np.log1p(c))
        states.append(np.column_stack(cols))
        counts.append(c)
    return states[0], states[1] - states[0], (counts[0] > 0) & (counts[1] > 0)


def _halves(events: EventArrays) -> tuple[float, float]:
    # A group's members may have no events at all: every row is then inactive
    # on ``current`` and the halves are never used.
    if not len(events.ts):
        return 0.0, 0.0
    t0 = float(events.ts.min())
    day_max = np.floor((float(events.ts.max()) - t0) / _DAY)
    return t0, t0 + ((day_max + 1) // 2) * _DAY


def _neighbours(
    ref: np.ndarray, query: np.ndarray, k: int, self_pos: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest reference rows of each query; ``self_pos`` leaves the query's own row out."""
    extra = 1 if self_pos is not None else 0
    k = min(k, len(ref) - extra)
    dist, idx = cKDTree(ref).query(query, k=k + extra, workers=-1)
    dist = np.asarray(dist).reshape(len(query), k + extra)
    idx = np.asarray(idx).reshape(len(query), k + extra)
    if self_pos is None:
        return dist, idx
    keep = idx != self_pos[:, None]
    keep[keep.all(axis=1), -1] = False
    return dist[keep].reshape(len(query), k), idx[keep].reshape(len(query), k)


def _mid_rank(sorted_values: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), 0.5)
    ok = np.flatnonzero(np.isfinite(x))
    if len(sorted_values) and len(ok):
        order = ok[np.argsort(x[ok], kind="stable")]  # sorted keys: cache-friendly searches
        lo = np.searchsorted(sorted_values, x[order], side="left")
        hi = np.searchsorted(sorted_values, x[order], side="right")
        out[order] = (lo + hi) / (2.0 * len(sorted_values))
    return out


def _sample(rng: np.random.Generator, rows: np.ndarray) -> np.ndarray:
    if len(rows) <= MAX_REFERENCE_ROWS:
        return rows
    return np.sort(rng.choice(rows, MAX_REFERENCE_ROWS, replace=False))


def _knn_coords(ref: Reference, x: np.ndarray) -> np.ndarray:
    assert ref.log_mu is not None and ref.log_sd is not None
    assert ref.pca_mean is not None and ref.pca_components is not None
    c = np.sign(x) * np.log1p(np.abs(x))
    coords: np.ndarray = ((c - ref.log_mu) / ref.log_sd - ref.pca_mean) @ ref.pca_components.T
    return coords


def _current(
    ref: Reference,
    events: EventArrays | None,
    n: int,
    self_pos: np.ndarray | None,
    states: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    out = np.full(n, np.nan)
    if events is None or ref.vel_s0 is None or ref.vel_u is None or ref.t_mid is None:
        return out
    s0, u, active = states if states is not None else event_states(events, n, ref.t_mid)
    if len(ref.vel_s0) < 2 or not active.any():
        return out
    rows = np.flatnonzero(active)
    pos = None if self_pos is None else self_pos[rows]
    if pos is not None and (pos < 0).all():
        pos = None
    if pos is None:
        _, idx = _neighbours(ref.vel_s0, s0[rows], VELOCITY_K, None)
        out[rows] = np.linalg.norm(u[rows] - ref.vel_u[idx].mean(axis=1), axis=1)
        return out
    member = pos >= 0
    if (~member).any():
        _, idx = _neighbours(ref.vel_s0, s0[rows[~member]], VELOCITY_K, None)
        out[rows[~member]] = np.linalg.norm(u[rows[~member]] - ref.vel_u[idx].mean(axis=1), axis=1)
    _, idx = _neighbours(ref.vel_s0, s0[rows[member]], VELOCITY_K, pos[member])
    out[rows[member]] = np.linalg.norm(u[rows[member]] - ref.vel_u[idx].mean(axis=1), axis=1)
    return out


def _knn(ref: Reference, x: np.ndarray, self_pos: np.ndarray | None) -> np.ndarray:
    assert ref.knn_ref is not None
    e = _knn_coords(ref, x)
    out = np.empty(len(x))
    member = np.zeros(len(x), dtype=bool) if self_pos is None else self_pos >= 0
    if (~member).any():
        dist, _ = _neighbours(ref.knn_ref, e[~member], KNN_K, None)
        out[~member] = dist.mean(axis=1)
    if member.any() and self_pos is not None:
        dist, _ = _neighbours(ref.knn_ref, e[member], KNN_K, self_pos[member])
        out[member] = dist.mean(axis=1)
    return out


def _axes(
    ref: Reference,
    x: np.ndarray,
    events: EventArrays | None,
    vel_pos: np.ndarray | None,
    knn_pos: np.ndarray | None,
    z: np.ndarray | None = None,
    states: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if z is None:
        z = to_gauss(ref.cdf_grid, ref.cdf_levels, x)
    r = z - ref.lw_location
    mahal = np.sqrt(np.maximum(np.sum((r @ ref.lw_precision) * r, axis=1), 0.0))
    values: dict[str, np.ndarray] = {"mahal": mahal}
    if "current" in ref.axes:
        values["current"] = _current(ref, events, len(x), vel_pos, states)
    if "knn" in ref.axes:
        values["knn"] = _knn(ref, x, knn_pos)
    return r.astype(np.float32), mahal.astype(np.float32), values


def _finish(
    ref: Reference,
    delta: np.ndarray,
    delta_norm: np.ndarray,
    values: dict[str, np.ndarray],
    sample_index: np.ndarray | None = None,
    without_current: bool = False,
    members: bool = True,
) -> RobustScores:
    """Fuse the axis ranks and calibrate. ``without_current``: the rows carry no
    events, so ``current`` is absent (its rank is NaN) and the rows are fused
    over the other axes and calibrated against ``cal_sorted_no_current``.
    ``members=False``: the rows were not in the fit, so their p-value is taken
    against the out-of-fold scores (when the reference has them): ``mahal``
    ranked among the out-of-fold values, fused, compared with the out-of-fold
    calibration. The score, axis ranks and rank percentile stay on the
    in-sample scale, so equal values score equally for members and new rows."""
    ranks = np.column_stack([_mid_rank(ref.axis_sorted[a], values[a]) for a in ref.axes])
    cal = ref.cal_sorted
    oof_cal = None if members else ref.cal_sorted_oof
    if without_current and ref.cal_sorted_no_current is not None:
        ranks[:, ref.axes.index("current")] = np.nan
        cal = ref.cal_sorted_no_current
        oof_cal = None if members else ref.cal_sorted_no_current_oof
    fused = np.nanmean(ranks, axis=1)
    if oof_cal is not None and ref.mahal_sorted_oof is not None:
        oof_ranks = ranks.copy()
        m = ref.axes.index("mahal")
        oof_ranks[:, m] = _mid_rank(ref.mahal_sorted_oof, values["mahal"])
        p = conformal_p(oof_cal, np.nanmean(oof_ranks, axis=1))
    else:
        p = conformal_p(cal, fused)
    return RobustScores(
        delta=delta,
        delta_norm=delta_norm,
        axis_values=values,
        axis_ranks=ranks.astype(np.float32),
        anomaly_score=fused,
        conformal_p=p,
        score_rank_pct=np.asarray(
            100.0 * np.searchsorted(cal, fused, side="left") / max(ref.n_fitted, 1),
            dtype=np.float64,
        ),
        sample_index=sample_index,
    )


@dataclass(frozen=True)
class Attribution:
    """Exact split of an entity's evidence ``bits = -log2(conformal_p)``.

    ``axis_bits`` sums to ``bits``: each present axis gets a share proportional
    to its marginal rarity ``-log2(1 - rank + 1/n)``. ``dim_bits`` splits the
    ``mahal`` share over the named dimensions by Shapley shares of the squared
    Mahalanobis distance (averaged over ``SHAPLEY_ORDERINGS`` orderings of the
    sequential decomposition), so it sums to ``axis_bits["mahal"]``.
    """

    bits: float
    axis_bits: dict[str, float]
    dim_bits: dict[str, float]


SHAPLEY_ORDERINGS = 32


def _shapley_d2(r: np.ndarray, cov: np.ndarray, orderings: int) -> np.ndarray:
    """Per-coordinate Shapley shares of ``r' cov^-1 r`` (sum = the squared distance).

    ``r`` is one row (D,) or many (n, D); the orderings are the same for every
    row, so each ordering's factor is computed once.
    """
    rows = np.atleast_2d(r)
    d = rows.shape[1]
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Only positive semi-definite (a constant or duplicated dimension): a
        # ridge far below the data's scale makes it factorable and leaves the
        # distance of rows in the covariance's range unchanged.
        scale = float(np.trace(cov)) / d if d else 1.0
        cov = cov + np.eye(d) * (1e-9 * (scale if scale > 0 else 1.0))
    rng = np.random.default_rng(0)
    out = np.zeros(rows.shape)
    for _ in range(orderings):
        perm = rng.permutation(d)
        chol = np.linalg.cholesky(cov[np.ix_(perm, perm)])
        out[:, perm] += np.linalg.solve(chol, rows[:, perm].T).T ** 2
    shares: np.ndarray = out / orderings
    return shares if r.ndim == 2 else shares[0]


def dim_shares_rows(ref: Reference, score_delta: np.ndarray) -> np.ndarray:
    """:func:`dim_shares` for many rows (n, D) at once; a row at zero distance
    gets zero shares rather than a uniform split."""
    r = np.asarray(score_delta, dtype=np.float64).reshape(-1, len(ref.lw_precision))
    if r.shape[0] == 0:
        return np.zeros(r.shape)
    cov = np.linalg.pinv(ref.lw_precision, hermitian=True)
    shares = _shapley_d2(r, cov, SHAPLEY_ORDERINGS)
    total = shares.sum(axis=1, keepdims=True)
    out: np.ndarray = np.divide(shares, total, out=np.zeros_like(shares), where=total > 0)
    return out


def dim_shares(
    ref: Reference, score_delta: np.ndarray, cov: np.ndarray | None = None
) -> np.ndarray:
    """Each score dimension's Shapley share of the entity's ``mahal`` evidence
    (non-negative, sums to 1). ``cov`` is ``pinv(ref.lw_precision)``, passed in
    when scoring many rows against one reference."""
    r = np.asarray(score_delta, dtype=np.float64)
    if cov is None:
        cov = np.linalg.pinv(ref.lw_precision, hermitian=True)
    shares = _shapley_d2(r, cov, SHAPLEY_ORDERINGS)
    total = float(shares.sum())
    out: np.ndarray = shares / total if total > 0 else np.full(len(r), 1.0 / len(r))
    return out


def attribute(
    ref: Reference, score_delta: np.ndarray, axis_ranks: np.ndarray, p: float
) -> Attribution:
    """Split one entity's evidence over its axes and, within ``mahal``, its dimensions.

    ``score_delta`` and ``axis_ranks`` are the entity's stored values (a NaN rank
    marks an absent axis, e.g. ``current`` for a row scored without events).
    """
    bits = float(-np.log2(p))
    ranks = np.asarray(axis_ranks, dtype=np.float64)
    present = [(a, float(r)) for a, r in zip(ref.axes, ranks, strict=True) if np.isfinite(r)]
    rarity = np.array([-np.log2(1.0 - r + 1.0 / max(ref.n_fitted, 1)) for _, r in present])
    weights = (
        rarity / rarity.sum() if rarity.sum() > 0 else np.full(len(present), 1.0 / len(present))
    )
    axis_bits = {a: bits * float(w) for (a, _), w in zip(present, weights, strict=True)}
    mahal_bits = axis_bits.get("mahal", 0.0)
    split = dim_shares(ref, score_delta)
    dim_bits = {d: mahal_bits * float(s) for d, s in zip(ref.dims, split, strict=True)}
    return Attribution(bits=bits, axis_bits=axis_bits, dim_bits=dim_bits)


@dataclass(frozen=True)
class Counterfactual:
    """One entity rescored with some score dimensions set to new values."""

    score_delta: np.ndarray  # (D,) rank-Gaussian deviation after the change
    axis_ranks: np.ndarray  # (A,) NaN for an absent axis
    anomaly_score: float
    conformal_p: float
    held_axes: tuple[str, ...]  # axes whose stored rank was kept


def counterfactual(
    ref: Reference,
    score_delta: np.ndarray,
    axis_ranks: np.ndarray,
    overrides: dict[int, float],
    *,
    members: bool = True,
) -> Counterfactual:
    """Rescore one entity with ``overrides`` (score-dim index → natural value).

    ``mahal`` is recomputed exactly from the changed rank-Gaussian coordinates;
    the other axes (motion, peers) need the entity's events or full row, so their
    stored ranks are kept. Without overrides the stored score is returned.
    ``members=False``: the entity was scored after the fit, so its p-value is
    taken against the out-of-fold calibration, as when it was stored.
    """
    r = np.asarray(score_delta, dtype=np.float64).copy()
    ranks = np.asarray(axis_ranks, dtype=np.float64).copy()
    # Stored ranks are float32; a mid-rank is k / (2m) exactly, so snap back to
    # that grid, or the fused score misses its own value in the reference.
    for a, axis in enumerate(ref.axes):
        grid = 2.0 * max(len(ref.axis_sorted[axis]), 1)
        if np.isfinite(ranks[a]):
            ranks[a] = np.round(ranks[a] * grid) / grid
    changed = False
    for d, value in overrides.items():
        z = to_gauss(ref.cdf_grid[d : d + 1], ref.cdf_levels, np.array([[float(value)]]))
        new = float(z[0, 0]) - float(ref.lw_location[d])
        # The stored coordinate is float32: a value it already holds is no change.
        if not np.isclose(new, r[d], rtol=1e-5, atol=1e-5):
            r[d] = new
            changed = True
    if changed:
        mahal = np.sqrt(max(float(r @ ref.lw_precision @ r), 0.0))
        ranks[ref.axes.index("mahal")] = _mid_rank(ref.axis_sorted["mahal"], np.array([mahal]))[0]
    no_current = (
        "current" in ref.axes
        and bool(np.isnan(ranks[ref.axes.index("current")]))
        and ref.cal_sorted_no_current is not None
    )
    cal = ref.cal_sorted_no_current if no_current else ref.cal_sorted
    oof_cal = ref.cal_sorted_no_current_oof if no_current else ref.cal_sorted_oof
    fused = float(np.nanmean(ranks))
    if not members and oof_cal is not None and ref.mahal_sorted_oof is not None:
        mahal = np.sqrt(max(float(r @ ref.lw_precision @ r), 0.0))
        oof_ranks = ranks.copy()
        oof_ranks[ref.axes.index("mahal")] = _mid_rank(ref.mahal_sorted_oof, np.array([mahal]))[0]
        p = float(conformal_p(oof_cal, np.array([np.nanmean(oof_ranks)]))[0])
    else:
        p = float(conformal_p(cal, np.array([fused]))[0])  # type: ignore[arg-type]
    return Counterfactual(
        score_delta=r.astype(np.float32),
        axis_ranks=ranks.astype(np.float32),
        anomaly_score=fused,
        conformal_p=p,
        held_axes=tuple(
            a for a in ref.axes if a != "mahal" and np.isfinite(ranks[ref.axes.index(a)])
        ),
    )


def fit_reference(
    x: np.ndarray,
    dims: Sequence[str],
    *,
    axes: Sequence[str] = AXES_DEFAULT,
    events: EventArrays | None = None,
    seed: int = 0,
) -> tuple[Reference, RobustScores]:
    """Fit the reference on the population ``x`` (n, D) and score the population itself.

    ``current`` is dropped when ``events`` is None. Rows drawn into a reference
    sample are scored with themselves left out; the others exactly as new rows.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    n = len(x)
    rng = np.random.default_rng(seed)
    used = tuple(a for a in axes if a != "current" or events is not None)
    levels = _levels(CDF_POINTS)
    grid, z = _sketch_and_gauss(x, None, levels)
    lw_location, lw_precision = ledoit_wolf(z)
    fields: dict[str, Any] = {}
    vel_pos = None
    knn_pos = None
    states = None
    sampled = []
    if "current" in used and events is not None:
        t0, t_mid = _halves(events)
        states = event_states(events, n, t_mid)
        s0, u, active = states
        rows = _sample(rng, np.flatnonzero(active))
        vel_pos = np.full(n, -1)
        vel_pos[rows] = np.arange(len(rows))
        sampled.append(rows)
        fields.update(t0=t0, t_mid=t_mid, vel_s0=s0[rows], vel_u=u[rows])
    if "knn" in used:
        c = np.sign(x) * np.log1p(np.abs(x))
        log_mu = c.mean(axis=0)
        log_sd = np.maximum(c.std(axis=0), 1e-9)
        pca_mean, pca_components = _principal_axes(
            (c - log_mu) / log_sd, min(KNN_DIMS, x.shape[1], n)
        )
        rows = _sample(rng, np.arange(n))
        knn_pos = np.full(n, -1)
        knn_pos[rows] = np.arange(len(rows))
        sampled.append(rows)
        coords = ((c - log_mu) / log_sd - pca_mean) @ pca_components.T
        fields.update(
            log_mu=log_mu,
            log_sd=log_sd,
            pca_mean=pca_mean,
            pca_components=pca_components,
            knn_ref=coords[rows],
        )
    partial = Reference(
        dims=tuple(dims),
        axes=used,
        cdf_grid=grid,
        cdf_levels=levels,
        lw_location=lw_location,
        lw_precision=lw_precision,
        axis_sorted={},
        cal_sorted=np.empty(0),
        n_fitted=n,
        **fields,
    )
    delta, delta_norm, values = _axes(partial, x, events, vel_pos, knn_pos, z, states)
    axis_sorted = {a: np.sort(v[np.isfinite(v)]) for a, v in values.items()}
    ranks = np.column_stack([_mid_rank(axis_sorted[a], values[a]) for a in used])
    others = [i for i, a in enumerate(used) if a != "current"]
    ref = replace(
        partial,
        axis_sorted=axis_sorted,
        cal_sorted=np.sort(ranks.mean(axis=1)),
        cal_sorted_no_current=(
            np.sort(ranks[:, others].mean(axis=1)) if "current" in used else None
        ),
    )
    ref = _with_out_of_fold(ref, x, ranks, seed)
    sample_index = np.unique(np.concatenate(sampled)) if sampled else None
    return ref, _finish(ref, delta, delta_norm, values, sample_index)


def _with_out_of_fold(ref: Reference, x: np.ndarray, ranks: np.ndarray, seed: int) -> Reference:
    """Add the calibration that rows unseen at fit time are compared with.

    A member's ``mahal`` comes from a fit that saw it, so it runs lower than a
    new row's. Each fold's ``mahal`` is recomputed with the CDF sketch and
    Ledoit–Wolf fitted on the other folds; new rows rank their ``mahal`` among
    these out-of-fold values, and the fused out-of-fold ranks are the
    calibration. ``current`` and ``knn`` keep their member values: those
    already leave the row out. The folds draw from their own generator, so the
    member samples above are untouched. Large populations (more than
    ``OOF_MAX_ROWS_PER_DIM`` rows per dimension) keep the in-sample calibration.
    """
    n = len(x)
    if n < 2 * OOF_FOLDS or n > OOF_MAX_ROWS_PER_DIM * x.shape[1]:
        return ref
    folds = np.random.default_rng(seed + 1).permutation(n) % OOF_FOLDS
    mahal = np.empty(n)
    for k in range(OOF_FOLDS):
        held = folds == k
        grid, z_fit = _sketch_and_gauss(x[~held], None, ref.cdf_levels)
        location, precision = ledoit_wolf(z_fit)
        r = to_gauss(grid, ref.cdf_levels, x[held]) - location
        mahal[held] = np.sqrt(np.maximum(np.sum((r @ precision) * r, axis=1), 0.0))
    # New rows rank their ``mahal`` among these out-of-fold values: ranked
    # against the in-sample values instead, the out-of-fold ones would pile up
    # above the in-sample maximum and make the top of the calibration a tie.
    mahal_oof_sorted = np.sort(mahal)
    oof = ranks.copy()
    m = ref.axes.index("mahal")
    oof[:, m] = _mid_rank(mahal_oof_sorted, mahal)
    others = [i for i, a in enumerate(ref.axes) if a != "current"]
    return replace(
        ref,
        mahal_sorted_oof=mahal_oof_sorted,
        cal_sorted_oof=np.sort(oof.mean(axis=1)),
        cal_sorted_no_current_oof=(
            np.sort(oof[:, others].mean(axis=1)) if "current" in ref.axes else None
        ),
    )


def score(
    ref: Reference,
    x: np.ndarray,
    events: EventArrays | None = None,
    *,
    members: bool = False,
) -> RobustScores:
    """Score rows against the frozen reference.

    Rows unseen at fit time (the default) are calibrated against the
    out-of-fold scores of the population. ``members=True`` rescores rows of the
    fitted population itself against the in-sample calibration, which
    reproduces their fitted p-values.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    delta, delta_norm, values = _axes(ref, x, events, None, None)
    return _finish(ref, delta, delta_norm, values, without_current=events is None, members=members)


_ARRAY_FIELDS = (
    "cdf_grid",
    "cdf_levels",
    "lw_location",
    "lw_precision",
    "cal_sorted",
    "vel_s0",
    "vel_u",
    "log_mu",
    "log_sd",
    "pca_mean",
    "pca_components",
    "knn_ref",
    "cal_sorted_no_current",
    "excluded_keys",
    "mahal_sorted_oof",
    "cal_sorted_oof",
    "cal_sorted_no_current_oof",
)


def reference_to_arrays(ref: Reference) -> dict[str, np.ndarray]:
    """Flat ``np.savez``-compatible arrays (no pickled objects)."""
    out: dict[str, np.ndarray] = {
        "dims": np.array(ref.dims, dtype=np.str_),
        "axes": np.array(ref.axes, dtype=np.str_),
        "n_fitted": np.array(ref.n_fitted, dtype=np.int64),
    }
    for name in _ARRAY_FIELDS:
        value = getattr(ref, name)
        if value is not None:
            out[name] = np.asarray(value)
    for name in ("t0", "t_mid"):
        value = getattr(ref, name)
        if value is not None:
            out[name] = np.array(value, dtype=np.float64)
    for axis, values in ref.axis_sorted.items():
        out[f"axis_sorted__{axis}"] = values
    return out


def reference_from_arrays(arrays: Mapping[str, np.ndarray]) -> Reference:
    axes = tuple(str(a) for a in arrays["axes"])
    optional = {name: arrays[name] for name in _ARRAY_FIELDS if name in arrays}
    return Reference(
        dims=tuple(str(d) for d in arrays["dims"]),
        axes=axes,
        axis_sorted={a: arrays[f"axis_sorted__{a}"] for a in axes},
        n_fitted=int(arrays["n_fitted"]),
        t0=float(arrays["t0"]) if "t0" in arrays else None,
        t_mid=float(arrays["t_mid"]) if "t_mid" in arrays else None,
        **optional,
    )


POOLED_GROUP = "(small groups)"
# Each group's reference is about 33 KB per dimension (the CDF sketch) plus its
# sorted calibration; past this many groups the stored and broadcast
# calibration grows past what one epoch should hold.
MAX_SCORE_GROUPS = 256


def min_reference_size(percentile: float) -> int:
    """Smallest group whose top member can be flagged at ``percentile``.

    A fitted member's smallest p-value is ``2 / (n + 1)`` (it counts itself),
    so a smaller group never flags anyone. Computed through :func:`is_anomaly`
    so the bound agrees with the flag's own floating-point comparison.
    """
    n = max(int(np.ceil(2.0 / max(1.0 - percentile / 100.0, 1e-12))) - 3, 1)
    while not is_anomaly(np.array([2.0 / (n + 1.0)]), percentile)[0]:
        n += 1
    return n


def pool_groups(
    ids: np.ndarray,
    percentile: float,
    min_group_size: int | None = None,
    *,
    stored_labels: bool = False,
) -> tuple[np.ndarray, frozenset[str]]:
    """Relabel members of groups below the reference size to ``POOLED_GROUP``.

    The size is ``min_reference_size(percentile)``, or ``min_group_size`` when
    that is larger; a smaller ``min_group_size`` would leave groups that can
    never flag, so it raises. Returns the labels and the pooled group values.
    A group already named ``POOLED_GROUP`` raises, unless ``stored_labels``:
    labels read back from a sphere may hold the pool itself.
    """
    bound = min_reference_size(percentile)
    if min_group_size is not None and min_group_size < bound:
        raise ValueError(
            f"min_group_size={min_group_size} is below {bound}, the smallest group "
            f"that can flag a member at the {percentile:g}th percentile"
        )
    threshold = max(bound, min_group_size or 0)
    ids = np.asarray(ids).astype(object)
    values, counts = np.unique(ids, return_counts=True)
    small = values[counts < threshold]
    if len(small) == 0:
        return ids, frozenset()
    if not stored_labels and POOLED_GROUP in values and set(small) - {POOLED_GROUP}:
        raise ValueError(
            f"a group is named {POOLED_GROUP!r}, the label small groups are pooled under; rename it"
        )
    out = ids.copy()
    out[np.isin(ids, small)] = POOLED_GROUP
    return out, frozenset(str(v) for v in small)


def subset_events(events: EventArrays | None, mask: np.ndarray) -> EventArrays | None:
    """The events of the rows in ``mask``, with ``code`` renumbered to the subset."""
    if events is None:
        return None
    new = np.cumsum(mask) - 1
    keep = mask[events.code]
    return EventArrays(
        new[events.code[keep]], events.direction[keep], events.ts[keep], events.amount[keep]
    )


@dataclass(frozen=True)
class GroupedReference:
    """A reference per group, with the population's for rows outside every group."""

    population: Reference  # unseen groups, and pooled rows when the pool is too small
    groups: dict[str, Reference]  # own groups, and POOLED_GROUP when it is large enough
    pooled: frozenset[str]  # group values folded into POOLED_GROUP

    def resolve(self, gid: str) -> tuple[Reference, str | None, str]:
        """``(reference, score group key or None, scope)`` for a row of group ``gid``."""
        key = POOLED_GROUP if gid in self.pooled else gid
        ref = self.groups.get(key)
        if ref is None:
            return self.population, None, "global"
        return ref, key, "pooled" if key == POOLED_GROUP else "group"


def fit_grouped_reference(
    x: np.ndarray,
    dims: Sequence[str],
    ids: np.ndarray,
    *,
    percentile: float,
    min_group_size: int | None = None,
    axes: Sequence[str] = AXES_DEFAULT,
    events: EventArrays | None = None,
    seed: int = 0,
    stored_labels: bool = False,
) -> tuple[GroupedReference, RobustScores, np.ndarray]:
    """Fit one reference per group (``ids`` per row) and the population's.

    Groups below the reference size are pooled (:func:`pool_groups`); the pool
    gets a reference when it reaches the size, otherwise its rows are scored
    with the population. Returns the references, every row's scores against
    its own reference and the per-row score group key (None: the population).
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    labels, pooled = pool_groups(ids, percentile, min_group_size, stored_labels=stored_labels)
    names, counts = np.unique(labels, return_counts=True)
    bound = max(min_reference_size(percentile), min_group_size or 0)
    own = [str(g) for g, c in zip(names, counts, strict=True) if c >= bound]
    if len(own) > MAX_SCORE_GROUPS:
        raise ValueError(
            f"{len(own)} score groups exceed {MAX_SCORE_GROUPS}: group by a coarser "
            "property or raise min_group_size"
        )
    population, fitted = fit_reference(x, dims, axes=axes, events=events, seed=seed)
    keys = np.full(len(x), None, dtype=object)
    groups: dict[str, Reference] = {}
    parts: list[tuple[np.ndarray, RobustScores]] = []
    for gid in own:
        mask = labels == gid
        ref, scores = fit_reference(
            x[mask], dims, axes=axes, events=subset_events(events, mask), seed=seed
        )
        groups[gid] = ref
        keys[mask] = gid
        parts.append((mask, scores))
    merged = {}
    for name in (
        "delta",
        "delta_norm",
        "axis_ranks",
        "anomaly_score",
        "conformal_p",
        "score_rank_pct",
    ):
        out = np.array(getattr(fitted, name), copy=True)
        for mask, scores in parts:
            out[mask] = getattr(scores, name)
        merged[name] = out
    grouped = GroupedReference(population=population, groups=groups, pooled=pooled)
    return grouped, replace(fitted, **merged, axis_values={}, sample_index=None), keys


def grouped_reference_to_arrays(grouped: GroupedReference) -> dict[str, np.ndarray]:
    """The population's arrays at the top level (what a reader unaware of groups
    loads), each group's under ``g<k>__`` keyed by index, not by value."""
    out = reference_to_arrays(grouped.population)
    names = sorted(grouped.groups)
    out["group_names"] = np.array(names, dtype=np.str_)
    out["pooled"] = np.array(sorted(grouped.pooled), dtype=np.str_)
    for k, gid in enumerate(names):
        for field, value in reference_to_arrays(grouped.groups[gid]).items():
            out[f"g{k}__{field}"] = value
    return out


def grouped_reference_from_arrays(arrays: Mapping[str, np.ndarray]) -> GroupedReference:
    return GroupedReference(
        population=reference_from_arrays(arrays),
        groups={
            str(gid): group_reference_from_arrays(arrays, k)
            for k, gid in enumerate(arrays["group_names"])
        },
        pooled=frozenset(str(g) for g in arrays["pooled"]),
    )


def group_reference_from_arrays(arrays: Mapping[str, np.ndarray], k: int) -> Reference:
    """The reference of the ``k``-th stored group; only its own members are read."""
    prefix = f"g{k}__"
    return reference_from_arrays(
        {name[len(prefix) :]: arrays[name] for name in arrays if name.startswith(prefix)}
    )
