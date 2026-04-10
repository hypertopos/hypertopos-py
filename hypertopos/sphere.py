# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import json
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import lance as _lance  # type: ignore[import-untyped]
import numpy as np

from hypertopos.engine.geometry import GDSEngine
from hypertopos.model.manifest import Contract, Manifest
from hypertopos.model.sphere import Sphere
from hypertopos.navigation.navigator import GDSNavigator
from hypertopos.storage.cache import GDSCache
from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter, _write_lance

if TYPE_CHECKING:
    from hypertopos.engine.forecast import ForecastProvider


class HyperSphere:
    def __init__(self, sphere: Sphere, reader: GDSReader,
                 writer: GDSWriter, cache: GDSCache) -> None:
        self._sphere = sphere
        self._reader = reader
        self._writer = writer
        self._cache = cache

    @classmethod
    def open(cls, base_path: str) -> HyperSphere:
        reader = GDSReader(base_path)
        writer = GDSWriter(base_path)
        cache = GDSCache()
        sphere = reader.read_sphere()
        return cls(sphere, reader, writer, cache)

    def session(self, agent_id: str) -> HyperSession:
        manifest = Manifest(
            manifest_id=str(uuid.uuid4()),
            agent_id=agent_id,
            snapshot_time=datetime.now(tz=UTC),
            status="active",
            line_versions={lid: line.current_version()
                           for lid, line in self._sphere.lines.items()},
            pattern_versions={pid: p.version
                              for pid, p in self._sphere.patterns.items()},
        )
        contract = Contract(
            manifest_id=manifest.manifest_id,
            pattern_ids=list(self._sphere.patterns.keys()),
        )
        # Create a session-scoped reader so each session holds its own
        # pinned Lance versions.  Construction is cheap — no I/O happens here.
        session_reader = GDSReader(str(self._reader._base))
        session_reader._storage_config = self._reader._storage_config
        # Inherit warm points cache from parent reader (PyArrow tables are
        # immutable and reference-counted — shallow copy is safe).
        session_reader._points_cache = dict(self._reader._points_cache)

        # Pin Lance dataset versions at session-open time (MVCC isolation).
        pinned: dict[str, int] = {}
        for pattern_id, version in manifest.pattern_versions.items():
            geo_path = (
                session_reader._base
                / "geometry" / pattern_id / f"v={version}" / "data.lance"
            )
            if geo_path.exists():
                with suppress(Exception):
                    pinned[pattern_id] = _lance.dataset(
                        str(geo_path)
                    ).latest_version
            tmp_path = (
                session_reader._base / "temporal" / pattern_id / "data.lance"
            )
            if tmp_path.exists():
                with suppress(Exception):
                    pinned[f"temporal:{pattern_id}"] = _lance.dataset(
                        str(tmp_path)
                    ).latest_version
            edge_path = (
                session_reader._base / "edges" / pattern_id / "data.lance"
            )
            if edge_path.exists():
                with suppress(Exception):
                    pinned[f"edges:{pattern_id}"] = _lance.dataset(
                        str(edge_path)
                    ).latest_version
        session_reader._pinned_lance_versions = pinned

        engine = GDSEngine(storage=session_reader, cache=self._cache)
        return HyperSession(manifest, contract, engine, session_reader, self._writer)


class HyperSession:
    def __init__(
        self,
        manifest: Manifest,
        contract: Contract,
        engine: GDSEngine,
        reader: GDSReader,
        writer: GDSWriter,
    ) -> None:
        self._manifest = manifest
        self._contract = contract
        self._engine = engine
        self._reader = reader
        self._writer = writer
        self._forecast_provider: ForecastProvider | None = None

    @property
    def forecast_provider(self) -> ForecastProvider | None:
        """Return the active forecast provider, or ``None`` for built-in."""
        return self._forecast_provider

    def set_forecast_provider(self, provider: ForecastProvider | None) -> None:
        """Set an external forecast provider (or ``None`` to revert to built-in)."""
        self._forecast_provider = provider

    def navigator(self) -> GDSNavigator:
        return GDSNavigator(
            engine=self._engine,
            storage=self._reader,
            manifest=self._manifest,
            contract=self._contract,
        )

    def recalibrate(
        self,
        pattern_id: str,
        soft_threshold: float | None = None,
        hard_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Full recalibration: recompute mu/sigma/theta, rebuild geometry, reset tracker.

        Reads all current shape vectors, recomputes population statistics,
        rebuilds delta vectors, overwrites geometry via Lance MVCC.
        """
        import pyarrow as pa

        from hypertopos.builder._stats import compute_stats
        from hypertopos.engine.calibration import CalibrationTracker
        from hypertopos.utils.arrow import delta_matrix_from_arrow

        sphere = self._reader.read_sphere()
        pattern = sphere.patterns[pattern_id]
        old_theta_norm = pattern.theta_norm
        version = self._manifest.pattern_versions[pattern_id]

        # 1. Read current geometry
        geo_table = self._reader.read_geometry(pattern_id, version)
        n = geo_table.num_rows

        # 2. Back-compute shape vectors from stored deltas
        deltas_old = delta_matrix_from_arrow(geo_table)
        sigma_old = np.maximum(pattern.sigma_diag, 1e-2)
        shape_vectors = deltas_old * sigma_old + pattern.mu

        # 3. Recompute stats
        new_mu, new_sigma, new_theta, new_deltas, new_delta_norms, _ = compute_stats(
            shape_vectors, anomaly_percentile=95.0,
        )

        # Apply higher sigma floor for prop_column dimensions (same as builder)
        prop_columns = getattr(pattern, 'prop_columns', [])
        if prop_columns:
            from hypertopos.builder._stats import SIGMA_EPS_PROP
            n_rel = len(pattern.relations)
            new_sigma[n_rel:] = np.maximum(new_sigma[n_rel:], SIGMA_EPS_PROP)
            new_deltas = ((shape_vectors - new_mu) / new_sigma).astype(np.float32)
            new_delta_norms = np.sqrt(
                np.einsum('ij,ij->i', new_deltas, new_deltas),
            ).astype(np.float32)
            D = shape_vectors.shape[1]
            theta_scalar = float(np.percentile(new_delta_norms, 95.0))
            component = theta_scalar / np.sqrt(D) if D > 0 else 0.0
            new_theta = np.full(D, component, dtype=np.float32)

        new_theta_norm = float(np.linalg.norm(new_theta))
        # Use >= (boundary inclusive) to match builder.py and geometry.py is_anomaly semantics.
        # Entities at exactly delta_norm == theta_norm are anomalous (on the boundary).
        is_anomaly = (new_theta_norm > 0.0) & (new_delta_norms >= new_theta_norm)

        # 4. Compute delta_rank_pct
        sorted_norms = np.sort(new_delta_norms)
        ranks = np.searchsorted(sorted_norms, new_delta_norms, side="left")
        delta_rank_pct = (ranks / max(n, 1) * 100).astype(np.float32)

        # 5. Rebuild geometry table — keep non-delta columns, replace delta columns
        d = new_deltas.shape[1]
        delta_list = pa.FixedSizeListArray.from_arrays(
            pa.array(new_deltas.ravel().tolist(), type=pa.float32()), d,
        )

        new_columns: dict[str, Any] = {}
        for col_name in geo_table.column_names:
            if col_name == "delta":
                new_columns["delta"] = delta_list
            elif col_name == "delta_norm":
                new_columns["delta_norm"] = pa.array(
                    new_delta_norms.tolist(), type=pa.float32(),
                )
            elif col_name == "is_anomaly":
                new_columns["is_anomaly"] = pa.array(
                    is_anomaly.tolist(), type=pa.bool_(),
                )
            elif col_name == "delta_rank_pct":
                new_columns["delta_rank_pct"] = pa.array(
                    delta_rank_pct.tolist(), type=pa.float32(),
                )
            elif col_name.startswith("delta_dim_"):
                dim_idx = int(col_name.split("_")[-1])
                if dim_idx < d:
                    new_columns[col_name] = pa.array(
                        new_deltas[:, dim_idx].tolist(), type=pa.float32(),
                    )
            else:
                new_columns[col_name] = geo_table.column(col_name)

        new_table = pa.table(new_columns)

        # 6. Lance overwrite + reindex
        geo_path = (
            self._reader._base / "geometry" / pattern_id / f"v={version}" / "data.lance"
        )
        _write_lance(new_table, str(geo_path), mode="overwrite")
        self._writer.build_index_if_needed(pattern_id, version)

        # 7. Update sphere.json
        sphere_path = self._reader._base / "_gds_meta" / "sphere.json"
        sphere_data = json.loads(sphere_path.read_text())
        pat_dict = sphere_data["patterns"][pattern_id]
        pat_dict["mu"] = new_mu.tolist()
        pat_dict["sigma_diag"] = new_sigma.tolist()
        pat_dict["theta"] = new_theta.tolist()
        pat_dict["last_calibrated_at"] = datetime.now(UTC).isoformat()
        pat_dict["population_size"] = n
        sphere_path.write_text(json.dumps(sphere_data, indent=2))

        # 8. Update geometry_stats cache
        self._writer.write_geometry_stats(
            pattern_id, version, new_delta_norms, new_theta_norm,
        )

        # 9. Invalidate temporal centroid cache (stale after recalibration)
        centroid_cache = (
            self._reader._base / "_gds_meta" / "temporal_centroids" / f"{pattern_id}.lance"
        )
        if centroid_cache.exists():
            import shutil
            shutil.rmtree(str(centroid_cache))

        # 10. Reset tracker
        old_tracker = self._reader.read_calibration_tracker(pattern_id)
        old_drift = old_tracker.drift_pct if old_tracker else 0.0
        s_thresh = (
            soft_threshold if soft_threshold is not None else (
                old_tracker.soft_threshold if old_tracker else 0.05
            )
        )
        h_thresh = (
            hard_threshold if hard_threshold is not None else (
                old_tracker.hard_threshold if old_tracker else 0.20
            )
        )
        tracker = CalibrationTracker.from_stats(
            new_mu, new_sigma, new_theta, n=n,
            soft_threshold=s_thresh, hard_threshold=h_thresh,
        )
        self._writer.write_calibration_tracker(pattern_id, tracker)

        return {
            "pattern_id": pattern_id,
            "previous_drift_pct": round(old_drift, 4),
            "new_theta_norm": round(new_theta_norm, 4),
            "old_theta_norm": round(old_theta_norm, 4),
            "records_recalibrated": n,
        }

    def close(self, purge_temporal: bool = False) -> None:
        self._manifest.status = "expired"
        if purge_temporal:
            self._writer.purge_agent_temporal(self._manifest.agent_id)

    def __enter__(self) -> HyperSession:
        return self

    def __exit__(self, *_) -> None:
        self.close()
