# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Welford online drift tracker for mu/sigma/theta calibration.

Tracks population drift after appends using Welford's online algorithm.
No I/O — pure numpy computation. Storage handled by reader/writer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

RESERVOIR_K: int = 10_000


@dataclass
class CalibrationTracker:
    calibrated_mu: np.ndarray
    calibrated_sigma: np.ndarray
    calibrated_theta: np.ndarray
    calibrated_n: int
    calibrated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    running_n: int = 0
    running_mean: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    running_m2: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    soft_threshold: float = 0.05
    hard_threshold: float = 0.20

    norm_reservoir: np.ndarray = field(
        default_factory=lambda: np.empty(RESERVOIR_K, dtype=np.float32),
    )
    norm_reservoir_count: int = 0
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(),
        repr=False,
    )

    @classmethod
    def from_stats(
        cls,
        mu: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray,
        n: int,
        soft_threshold: float = 0.05,
        hard_threshold: float = 0.20,
    ) -> CalibrationTracker:
        return cls(
            calibrated_mu=mu.copy(),
            calibrated_sigma=sigma.copy(),
            calibrated_theta=theta.copy(),
            calibrated_n=n,
            running_n=0,
            running_mean=mu.copy(),
            running_m2=np.zeros_like(mu),
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            norm_reservoir=np.empty(RESERVOIR_K, dtype=np.float32),
            norm_reservoir_count=0,
        )

    def update(self, shape_vectors: np.ndarray) -> None:
        """Welford batch update with new shape vectors. O(k * d)."""
        for x in shape_vectors:
            self.running_n += 1
            delta = x - self.running_mean
            self.running_mean = self.running_mean + delta / self.running_n
            delta2 = x - self.running_mean
            self.running_m2 = self.running_m2 + delta * delta2

    def update_norms(self, batch_norms: np.ndarray) -> None:
        """Reservoir sampling to maintain up to RESERVOIR_K norm samples.

        Used for approximate theta estimation without full geometry scan.
        """
        for norm in batch_norms:
            if self.norm_reservoir_count < RESERVOIR_K:
                self.norm_reservoir[self.norm_reservoir_count] = norm
                self.norm_reservoir_count += 1
            else:
                j = int(self._rng.integers(0, self.norm_reservoir_count + 1))
                if j < RESERVOIR_K:
                    self.norm_reservoir[j] = norm
                self.norm_reservoir_count += 1

    @property
    def drift_pct(self) -> float:
        """Drift: ||running_mean - calibrated_mu|| / ||calibrated_sigma||."""
        if self.running_n == 0:
            return 0.0
        shift = self.running_mean - self.calibrated_mu
        return float(np.linalg.norm(shift) / np.linalg.norm(self.calibrated_sigma))

    @property
    def is_stale(self) -> bool:
        return self.drift_pct > self.soft_threshold

    @property
    def is_blocked(self) -> bool:
        return self.drift_pct > self.hard_threshold

    def reset(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        theta: np.ndarray,
        n: int,
    ) -> None:
        """Reset tracker after recalibration."""
        self.calibrated_mu = mu.copy()
        self.calibrated_sigma = sigma.copy()
        self.calibrated_theta = theta.copy()
        self.calibrated_n = n
        self.calibrated_at = datetime.now(UTC)
        self.running_n = 0
        self.running_mean = mu.copy()
        self.running_m2 = np.zeros_like(mu)
        self.norm_reservoir = np.empty(RESERVOIR_K, dtype=np.float32)
        self.norm_reservoir_count = 0
