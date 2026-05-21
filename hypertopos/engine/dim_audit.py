# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Per-dimension signal-vs-label audit for anchor patterns.

Given a labelled subset of entities, compute per-dim `AUROC(|delta_i|,
label)` to surface:
    - signal carriers (AUROC > upper_threshold)
    - anti-signal dims (AUROC < lower_threshold) that DRAG the L2-norm
      ranking AWAY from the label
    - neutral dims (no signal)

Diagnostic tool — surfaces whether the polygon's dim selection is
leaving discriminative signal on the table. The filtered-polygon
reconstruction (drop anti-signal dims, or keep only signal-carriers)
often raises `delta_norm` AUROC measurably.

Does NOT change sphere format. Does NOT bake labels into the pattern.
Caller supplies labels at audit time; output is a per-dim score table
+ a recommended mask for downstream filtered scoring.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "compute_per_dim_label_auroc",
    "filter_delta_norm",
    "fit_lda_direction",
    "normality_test_per_dim",
]

# Shapiro-Wilk is the highest-power normality test for small N but its
# scipy implementation rejects N > 5000. Above that, Kolmogorov-Smirnov
# against a fitted normal is used — same null hypothesis, much weaker
# power at small N but well-defined at any N.
_SHAPIRO_MAX_N = 5000


def _auroc_unsafe(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via the rank-based identity:
        AUROC = (sum_rank_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Avoids sklearn dependency on the engine layer. Returns 0.5 when one
    class is missing or when all scores are tied.
    """
    n = labels.size
    if n == 0:
        return 0.5
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    if np.std(scores) < 1e-12:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average ranks for ties (matters when many equal scores).
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
        i = j
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_per_dim_label_auroc(
    *,
    deltas: np.ndarray,
    labels: np.ndarray,
    dim_labels: list[str] | None = None,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
) -> dict[str, Any]:
    """Per-dim signal audit against a binary label vector.

    For each dim ``i``, compute ``AUROC(|delta[:, i]|, labels)`` — how
    well the dim's absolute-deviation magnitude alone discriminates
    positive from negative entities.

    Args:
        deltas: ``(n_entities, n_dims)`` delta vectors as returned by
            geometry. Rows are entities, columns are pattern dims.
        labels: ``(n_entities,)`` binary labels {0, 1}.
        dim_labels: optional per-dim names (length n_dims). When None,
            uses ``f"dim_{i}"``.
        upper_threshold: AUROC threshold for signal-carrier
            classification (default 0.55 — adjust for noisy domains).
        lower_threshold: AUROC threshold for anti-signal classification
            (below this → dim is reverse-correlated with the label).

    Returns:
        ``{per_dim: [{idx, label, auroc, classification}, ...],
        n_signal, n_neutral, n_anti, signal_mask: ndarray[bool],
        signal_idx: list[int], anti_idx: list[int]}``.
        ``signal_mask[i] = True`` iff dim ``i`` is a signal carrier.
    """
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be 2-D, got shape {deltas.shape}")
    if labels.shape != (deltas.shape[0],):
        raise ValueError(
            f"labels shape mismatch: {labels.shape} vs "
            f"({deltas.shape[0]},)",
        )
    n_dims = deltas.shape[1]
    if dim_labels is None:
        dim_labels = [f"dim_{i}" for i in range(n_dims)]
    if len(dim_labels) != n_dims:
        raise ValueError(
            f"dim_labels length {len(dim_labels)} != n_dims {n_dims}",
        )

    per_dim: list[dict[str, Any]] = []
    signal_mask = np.zeros(n_dims, dtype=bool)
    signal_idx: list[int] = []
    anti_idx: list[int] = []
    n_signal = n_anti = n_neutral = 0
    for i in range(n_dims):
        abs_d = np.abs(deltas[:, i])
        a = _auroc_unsafe(abs_d, labels)
        if a > upper_threshold:
            cls = "signal"
            n_signal += 1
            signal_mask[i] = True
            signal_idx.append(i)
        elif a < lower_threshold:
            cls = "anti"
            n_anti += 1
            anti_idx.append(i)
        else:
            cls = "neutral"
            n_neutral += 1
        per_dim.append({
            "idx": i,
            "label": dim_labels[i],
            "auroc": a,
            "classification": cls,
        })
    return {
        "per_dim": per_dim,
        "n_signal": n_signal,
        "n_neutral": n_neutral,
        "n_anti": n_anti,
        "signal_mask": signal_mask,
        "signal_idx": signal_idx,
        "anti_idx": anti_idx,
    }


def filter_delta_norm(
    *,
    deltas: np.ndarray,
    keep_idx: list[int] | np.ndarray,
) -> np.ndarray:
    """Recompute L2 norm of deltas restricted to ``keep_idx`` dims.

    Use the ``signal_idx`` from ``compute_per_dim_label_auroc`` to drop
    anti-signal + neutral dims and recompute the polygon's anomaly
    ranking.

    Returns:
        ``(n_entities,)`` array of filtered delta_norm values.
    """
    keep = np.asarray(list(keep_idx), dtype=np.intp)
    if keep.size == 0:
        return np.zeros(deltas.shape[0], dtype=np.float64)
    return np.linalg.norm(deltas[:, keep], axis=1)


def fit_lda_direction(
    *,
    deltas: np.ndarray,
    labels: np.ndarray,
    regularization: float = 1e-6,
) -> dict[str, Any]:
    """Fit a Fisher LDA direction over labelled delta vectors.

    Solves the regularised Fisher discriminant
    ``w = (S_w + reg*I)^{-1} (mu_anom - mu_normal)`` and returns the
    unit-normalised direction together with its Fisher discriminant
    ratio. The direction is sign-oriented so that
    ``w . (mu_anom - mu_normal) > 0`` — projecting onto ``w`` lands the
    anomalous class on the positive side, deterministic across rebuilds.

    Within-class scatter is ridge-regularised by ``regularization * I``
    so the linear solve stays numerically stable on rank-deficient
    ``S_w`` (typical when ``N`` is small relative to ``D``).

    Args:
        deltas: ``(n_entities, n_dims)`` delta vectors. Rows are entities,
            columns are pattern dims.
        labels: ``(n_entities,)`` binary labels {0, 1}.
        regularization: ridge added to the within-class scatter matrix
            before solving. Default ``1e-6`` matches the engine-wide
            epsilon style. Must be non-negative.

    Returns:
        ``{direction: ndarray(n_dims,), fisher_score: float, n_anom: int,
        n_normal: int}``. ``direction`` has unit L2 norm and the sign
        convention above; ``fisher_score`` is the Fisher discriminant
        ratio ``(w . diff)^2 / (w . S_w_reg . w)``.

    Raises:
        ValueError: if ``deltas`` is not 2-D, if ``labels`` shape does
            not match, if labels are not binary {0, 1}, if either class
            is missing, if either class has fewer than 2 samples, if
            ``deltas`` contains non-finite values, or if the two class
            means are identical (no discriminative direction exists).
    """
    if deltas.ndim != 2:
        raise ValueError(f"deltas must be 2-D, got shape {deltas.shape}")
    if labels.shape != (deltas.shape[0],):
        raise ValueError(
            f"labels shape mismatch: {labels.shape} vs "
            f"({deltas.shape[0]},)",
        )
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            f"labels must be binary 0/1, got values {unique_labels.tolist()}",
        )
    if not np.isfinite(deltas).all():
        raise ValueError("deltas contain NaN or inf")

    mask_anom = labels == 1
    mask_normal = labels == 0
    n_anom = int(mask_anom.sum())
    n_normal = int(mask_normal.sum())
    if n_anom == 0 or n_normal == 0:
        raise ValueError(
            f"both classes must be present: n_anom={n_anom}, "
            f"n_normal={n_normal}",
        )
    if n_anom < 2 or n_normal < 2:
        raise ValueError(
            f"each class needs >=2 samples for within-class scatter: "
            f"n_anom={n_anom}, n_normal={n_normal}",
        )

    mu_anom = deltas[mask_anom].mean(axis=0)
    mu_normal = deltas[mask_normal].mean(axis=0)
    if np.allclose(mu_anom, mu_normal):
        raise ValueError("class means identical — no LDA direction")
    diff = mu_anom - mu_normal

    x_a = deltas[mask_anom] - mu_anom
    x_n = deltas[mask_normal] - mu_normal
    s_w = (x_a.T @ x_a + x_n.T @ x_n) / (n_anom + n_normal - 2)

    n_dims = deltas.shape[1]
    s_w_reg = s_w + regularization * np.eye(n_dims)

    w = np.linalg.solve(s_w_reg, diff)

    # Sign-orient so w . diff > 0 — positive projection on anomalous side.
    if w @ diff < 0:
        w = -w

    # Unit-normalise.
    w_norm = np.linalg.norm(w)
    w = w / max(w_norm, 1e-12)

    fisher_score = float(
        (w @ diff) ** 2 / max(w @ s_w_reg @ w, 1e-12),
    )

    return {
        "direction": w,
        "fisher_score": fisher_score,
        "n_anom": n_anom,
        "n_normal": n_normal,
    }


def normality_test_per_dim(values: np.ndarray) -> dict[str, Any]:
    """Test whether a 1-D sample is drawn from a normal distribution.

    Hypothesis pair is the standard one for both tests:
        H0: the sample is normally distributed
        H1: the sample is NOT normally distributed

    The test is selected by sample size — Shapiro-Wilk has the highest
    power for small N but its scipy implementation rejects N > 5000;
    above that threshold we switch to a Kolmogorov-Smirnov test against
    a normal fitted to the sample mean and (unbiased) standard
    deviation. Both tests reject H0 when p < alpha (typically 0.01 or
    0.05) — a low p-value means the dim distribution is NOT normal, so
    the gaussian z-score `(x - mu) / sigma` is a poor anomaly scorer on
    that dim.

    NaN values are stripped before testing; the test runs on the finite
    subset. Less than three finite values, or zero variance after NaN
    drop, returns ``p_value = nan`` and the caller treats that as
    "insufficient data — no claim".

    Args:
        values: 1-D numeric array. NaN entries are dropped. Need not be
            standardised; KS standardises internally against the sample
            mean and std.

    Returns:
        ``{test_name: "shapiro" | "ks", statistic: float, p_value: float,
        n: int}``. ``n`` is the count AFTER NaN drop. ``p_value`` is in
        ``[0, 1]`` when defined, ``nan`` when the sample is too small or
        constant.

    Raises:
        ValueError: if ``values`` is not 1-D.
    """
    from scipy import stats

    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)

    # Both tests need at least three observations and non-zero spread.
    # Returning nan rather than raising keeps the caller's per-dim loop
    # robust against pathological columns (all-zero, all-NaN, constant)
    # without forcing it to pre-filter — the warning emitter already
    # ignores nan p-values.
    if n < 3 or float(np.std(finite, ddof=1)) < 1e-12:
        return {
            "test_name": "shapiro" if n <= _SHAPIRO_MAX_N else "ks",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n": n,
        }

    if n <= _SHAPIRO_MAX_N:
        result = stats.shapiro(finite)
        return {
            "test_name": "shapiro",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
        }

    # KS against a fitted normal — standardise the sample and compare
    # against the standard normal CDF. Equivalent to kstest(values,
    # 'norm', args=(mean, std)) but lets scipy fall on the well-tested
    # one-sample path.
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1))
    standardised = (finite - mean) / std
    result = stats.kstest(standardised, "norm")
    return {
        "test_name": "ks",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": n,
    }
