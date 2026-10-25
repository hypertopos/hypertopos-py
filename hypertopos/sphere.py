# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import uuid
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hypertopos.engine.geometry import GDSEngine, calibrated_deltas, shapes_from_deltas
from hypertopos.model.manifest import Contract, Manifest
from hypertopos.model.sphere import Sphere
from hypertopos.navigation.navigator import GDSNavigator
from hypertopos.storage import _lance_branches
from hypertopos.storage._lance_branches import BranchInfo, Pin, PromoteReport
from hypertopos.storage.cache import GDSCache
from hypertopos.storage.exceptions import GDSError, GDSStorageError
from hypertopos.storage.reader import GDSReader
from hypertopos.storage.writer import GDSWriter, write_sphere_json

if TYPE_CHECKING:
    from hypertopos.engine.forecast import ForecastProvider


class HyperSphere:
    def __init__(
        self, sphere: Sphere, reader: GDSReader, writer: GDSWriter, cache: GDSCache
    ) -> None:
        self._sphere = sphere
        self._reader = reader
        self._writer = writer
        self._cache = cache

    @classmethod
    def open(cls, base_path: str, branch: str | None = None) -> HyperSphere:
        """Open a sphere; with ``branch``, sessions default to that branch."""
        if branch:
            _lance_branches.verify_branch(base_path, branch)
        reader = GDSReader(base_path, branch=branch)
        writer = GDSWriter(base_path)
        cache = GDSCache()
        sphere = reader.read_sphere()
        return cls(sphere, reader, writer, cache)

    def session(self, agent_id: str, branch: str | None = None) -> HyperSession:
        """Open a session pinned to the current state of ``branch``.

        ``branch=None`` uses the branch the sphere was opened on, or main when
        it was opened without one; open the sphere without ``branch`` to get
        main sessions.
        """
        ref = branch or self._reader.branch
        if ref and ref != self._reader.branch:
            _lance_branches.verify_branch(self._reader._base, ref)
        # Create a session-scoped reader so each session holds its own
        # pinned Lance versions.  Construction is cheap — no I/O happens here.
        session_reader = GDSReader(str(self._reader._base), branch=ref)
        session_reader._storage_config = self._reader._storage_config
        # Inherit warm points cache from parent reader (PyArrow tables are
        # immutable and reference-counted — shallow copy is safe).
        session_reader._points_cache = dict(self._reader._points_cache)

        # Pin sphere.json together with the Lance dataset versions (MVCC
        # isolation): the session reads this one snapshot until it refreshes.
        session_reader.take_snapshot()

        sphere = self._sphere if ref == self._reader.branch else session_reader.read_sphere()
        manifest = Manifest(
            manifest_id=str(uuid.uuid4()),
            agent_id=agent_id,
            snapshot_time=datetime.now(tz=UTC),
            status="active",
            line_versions={lid: line.current_version() for lid, line in sphere.lines.items()},
            pattern_versions={pid: p.version for pid, p in sphere.patterns.items()},
        )
        contract = Contract(
            manifest_id=manifest.manifest_id,
            pattern_ids=list(sphere.patterns.keys()),
        )
        # The polygon cache is keyed by snapshot, so main and branch sessions
        # can share it.
        engine = GDSEngine(storage=session_reader, cache=self._cache)
        return HyperSession(manifest, contract, engine, session_reader, self._writer)

    def create_branch(self, name: str, *, created_by: str = "", purpose: str = "") -> BranchInfo:
        """Create branch ``name`` from the current main state (no data copied)."""
        _lance_branches.create_branch(
            self._reader._base, name, created_by=created_by, purpose=purpose
        )
        return next(b for b in self.branches() if b.name == name)

    def branches(self) -> list[BranchInfo]:
        """Branches of this sphere, sorted by name."""
        return _lance_branches.list_branches(self._reader._base)

    def delete_branch(self, name: str) -> None:
        """Delete branch ``name``: its Lance branches and its metadata overlay.

        Refused while an open (not closed) session in this process reads the
        branch; close those sessions first.
        """
        _refuse_if_branch_in_use(self._reader._base, name)
        _lance_branches.delete_branch(self._reader._base, name)

    def promote_branch(self, name: str, *, agent_id: str = "_promote") -> PromoteReport:
        """Make main equal to branch ``name`` by replaying its recipes on main.

        Refused unless main is still at the state the branch was created from
        and every branch commit is recorded by a recipe. Each replayed
        recalibration must reproduce the branch's fit exactly (sha256 of mu);
        on a mismatch or an error main's datasets and metadata are restored,
        the branch is kept and ``GDSError`` is raised. On success the branch
        is deleted, so promotion is refused while an open session in this
        process reads the branch.
        """
        base = self._reader._base
        _refuse_if_branch_in_use(base, name)
        # Always promote_lock first, then the branch's recipe lock: an append
        # waits for the promotion instead of landing in a log already replayed.
        with _lance_branches.promote_lock(base), _lance_branches.recipe_lock(base, name):
            info = _lance_branches.check_promotable(base, name)
            recipes = sorted(info["recipes"], key=lambda r: r["seq"])
            replayable_ops = {"recalibrate", "incremental_update", "finalize_incremental"}
            for recipe in recipes:
                if recipe["op"] not in replayable_ops:
                    raise GDSError(f"branch {name!r}: cannot replay recipe op {recipe['op']!r}")
            pattern_ids = sorted({r["pattern_id"] for r in recipes})
            written = sorted({key for r in recipes for key in r.get("lance_version_after", {})})
            before = _lance_branches.main_versions(base, written)
            _lance_branches.backup_main_metadata(base, name, pattern_ids)
            # Replay on main even when this sphere was opened on a branch:
            # its sessions default to that branch.
            main = HyperSphere.open(str(base)) if self._reader.branch else self
            try:
                with main.session(agent_id) as session:
                    for recipe in recipes:
                        if recipe["op"] != "recalibrate":
                            main._replay_ingest(name, recipe)
                            session.refresh()
                            continue
                        out = session.recalibrate(
                            recipe["pattern_id"],
                            recipe["soft_threshold"],
                            recipe["hard_threshold"],
                            exclude_keys=recipe["exclude_keys"] or None,
                        )
                        expected = recipe["result"]
                        theta_norm = session.sphere().patterns[recipe["pattern_id"]].theta_norm
                        if (
                            out["mu_sha256"] != expected["mu_sha256"]
                            or out["records_recalibrated"] != expected["population_size"]
                            or not math.isclose(theta_norm, expected["theta_norm"], rel_tol=1e-9)
                        ):
                            raise GDSError(
                                f"replay of branch {name!r} diverged at recipe "
                                f"{recipe['seq']}: main produced {out['mu_sha256']} for "
                                f"{out['records_recalibrated']} rows, the branch recorded "
                                f"{expected['mu_sha256']} for {expected['population_size']}"
                            )
            except BaseException:
                _lance_branches.restore_main_versions(base, before)
                _lance_branches.restore_main_metadata(base, name)
                raise
            after = _lance_branches.main_versions(base, written)
            _lance_branches.delete_branch(base, name)
        self._sphere = self._reader.read_sphere()
        return PromoteReport(name, len(recipes), before, after)

    def _replay_ingest(self, name: str, recipe: dict[str, Any]) -> None:
        """Replay an ``incremental_update`` / ``finalize_incremental`` recipe on main."""
        import pyarrow as pa

        from hypertopos.builder.builder import GDSBuilder

        base = self._reader._base
        builder = GDSBuilder(self._sphere.sphere_id, str(base))
        pid = recipe["pattern_id"]
        if recipe["op"] == "finalize_incremental":
            builder.finalize_incremental(pid)
            return
        points = None
        if recipe["points_file"]:
            path = _lance_branches.overlay_dir(Path(base).resolve(), name) / recipe["points_file"]
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != recipe["points_sha256"]:
                raise GDSError(
                    f"replay of branch {name!r} diverged at recipe {recipe['seq']}: "
                    f"the stored rows {recipe['points_file']} changed"
                )
            with pa.memory_map(str(path)) as source:
                points = pa.ipc.open_file(source).read_all()
        # An automatic recalibration on the branch is a recipe of its own.
        out = builder.incremental_update(
            pid,
            changed_entities=points,
            deleted_keys=recipe["deleted_keys"] or None,
            recalibrate="never",
            reindex=recipe["reindex"],
            recompute_ranks=recipe["recompute_ranks"],
        )
        got = {
            "added": out.added,
            "modified": out.modified,
            "deleted": out.deleted,
            "population_size": out.population_size,
        }
        if got != recipe["result"]:
            raise GDSError(
                f"replay of branch {name!r} diverged at recipe {recipe['seq']}: "
                f"main produced {got}, the branch recorded {recipe['result']}"
            )


@dataclass(frozen=True)
class SnapshotStatus:
    """Whether a newer sphere state exists than the one a session reads."""

    pinned_at: datetime
    stale: bool
    metadata_changed: bool
    pinned_versions: dict[str, Pin]
    latest_versions: dict[str, Pin]


# Sessions alive in this process; weak, so an unclosed session that is
# garbage-collected stops counting as a branch reader.
_LIVE_SESSIONS: weakref.WeakSet[HyperSession] = weakref.WeakSet()


def _refuse_if_branch_in_use(base: Path | str, name: str) -> None:
    """Raise when an open session in this process reads branch ``name`` of ``base``."""
    target = Path(base).resolve()
    holders = [
        s._manifest.agent_id
        for s in list(_LIVE_SESSIONS)
        if s._reader.branch == name
        and s._manifest.status != "expired"
        and Path(s._reader._base).resolve() == target
    ]
    if holders:
        raise GDSError(
            f"branch {name!r} is read by {len(holders)} open session(s) in this process "
            f"(agent ids: {sorted(holders)}); close them first"
        )


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
        _LIVE_SESSIONS.add(self)

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

    def sphere(self) -> Sphere:
        """The sphere metadata of the session's snapshot."""
        return self._reader.read_sphere()

    def refresh(self) -> None:
        """Move the session to the latest sphere state.

        Navigators created from this session share its reader and manifest,
        so they read the new snapshot too.
        """
        self._reader._repin()
        sphere = self._reader.read_sphere()
        self._manifest.line_versions.clear()
        self._manifest.line_versions.update(
            {lid: line.current_version() for lid, line in sphere.lines.items()}
        )
        self._manifest.pattern_versions.clear()
        self._manifest.pattern_versions.update(
            {pid: p.version for pid, p in sphere.patterns.items()}
        )

    def is_stale(self) -> SnapshotStatus:
        """Compare the session's snapshot with the sphere's current state."""
        reader = self._reader
        pinned = dict(reader._pinned_lance_versions)
        latest = reader.latest_lance_versions(
            json.loads(reader.sphere_json_text()).get("patterns", {})
        )
        metadata_changed = reader.read_meta_files() != reader._meta_files
        return SnapshotStatus(
            pinned_at=reader._snapshot_at or self._manifest.snapshot_time,
            stale=metadata_changed or latest != pinned,
            metadata_changed=metadata_changed,
            pinned_versions=pinned,
            latest_versions=latest,
        )

    def recalibrate(
        self,
        pattern_id: str,
        soft_threshold: float | None = None,
        hard_threshold: float | None = None,
        *,
        branch: str | None = None,
        exclude_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Full recalibration: recompute mu/sigma/theta, rebuild geometry, reset tracker.

        Reads all current shape vectors, recomputes population statistics,
        rebuilds delta vectors, and replaces the calibration-dependent geometry
        columns in one Lance commit (key and relation columns and their indexes
        are left as stored).

        ``exclude_keys`` leaves those entities out of the fit; every entity is
        still scored against it. ``branch`` on a main session runs the
        recalibration on that branch (created when absent) and leaves main and
        this session untouched; on a branch session it must name the session's
        branch. A recalibration on a branch is recorded in its recipe log.
        """
        if isinstance(exclude_keys, str):
            raise TypeError("exclude_keys must be a collection of keys, not a string")
        if branch is not None and branch != self._reader.branch:
            if self._reader.branch:
                raise ValueError(
                    f"branch {branch!r} does not match this session's branch "
                    f"{self._reader.branch!r}"
                )
            hs = HyperSphere.open(str(self._reader._base))
            created = branch not in _lance_branches.branch_names(self._reader._base)
            if created:
                hs.create_branch(branch, created_by=self._manifest.agent_id)
            try:
                with hs.session(self._manifest.agent_id, branch=branch) as branch_session:
                    return branch_session.recalibrate(
                        pattern_id, soft_threshold, hard_threshold, exclude_keys=exclude_keys
                    )
            except BaseException:
                # A branch this call created and could not fill would block
                # build() and show up in branches(); remove it.
                if created:
                    hs.delete_branch(branch)
                raise
        import pyarrow as pa

        from hypertopos.builder._stats import (
            compute_per_dim_anomaly_count,
            compute_stats,
            per_dim_anomaly_thresholds,
        )
        from hypertopos.builder._theta_sensitivity import compute_theta_sensitivity_from_sorted
        from hypertopos.engine.calibration import CalibrationTracker
        from hypertopos.storage.calibration_history import (
            history_dir,
            write_calibration_history_epoch,
        )
        from hypertopos.utils.arrow import delta_matrix_from_arrow

        sphere = self._reader.read_sphere()
        pattern = sphere.patterns[pattern_id]
        old_theta_norm = pattern.theta_norm
        version = self._manifest.pattern_versions[pattern_id]

        # 1. Read the pinned geometry fragment by fragment: only the columns the
        # fit needs, with row addresses so step 6 can replace columns in place.
        geo_path = self._reader._base / "geometry" / pattern_id / "data.lance"
        pinned_ds = self._reader.dataset(geo_path, key=pattern_id)
        geo_columns = pinned_ds.schema.names
        parts = [
            fragment.to_table(columns=["primary_key", "delta"], with_row_id=True)
            for fragment in pinned_ds.get_fragments()
        ]
        row_ids = [part["_rowid"] for part in parts]
        geo_table = (
            pa.concat_tables([part.drop_columns(["_rowid"]) for part in parts])
            if parts
            else self._reader.read_geometry(pattern_id, version, columns=["primary_key", "delta"])
        )
        n = geo_table.num_rows

        # 2. Back-compute shape vectors from stored deltas
        deltas_old = delta_matrix_from_arrow(geo_table)
        shape_vectors = shapes_from_deltas(deltas_old, pattern)
        excluded = sorted(set(exclude_keys or ()))
        fit_mask = np.ones(n, dtype=bool)
        if excluded:
            keys = geo_table.column("primary_key").to_pylist()
            unknown = sorted(set(excluded) - set(keys))
            if unknown:
                raise ValueError(f"exclude_keys not in pattern '{pattern_id}': {unknown[:10]}")
            fit_mask = ~np.isin(np.asarray(keys, dtype=object), excluded)
            if not fit_mask.any():
                raise ValueError(
                    f"exclude_keys leaves no entity left to fit pattern '{pattern_id}'"
                )

        # 3. Recompute stats in the pattern's metric: a Mahalanobis pattern
        # gets a fresh cholesky_inv; stored dimension_weights are kept as-is.
        # The anomaly percentile is the one the pattern was built with
        # (spheres that predate the stored value use the 95.0 default).
        meta_root = self._reader.meta_root
        sphere_path = meta_root / "_gds_meta" / "sphere.json"
        pct = float(
            json.loads(self._reader.sphere_json_text())["patterns"][pattern_id].get(
                "anomaly_percentile", 95.0
            )
        )
        new_mu, new_sigma, _, _, _, new_cholesky_inv = compute_stats(
            shape_vectors[fit_mask],
            anomaly_percentile=pct,
            use_mahalanobis=pattern.cholesky_inv is not None,
        )

        # Apply higher sigma floor for prop_column dimensions (same slice as
        # the builder: right after relations and event dimensions).
        prop_columns = getattr(pattern, "prop_columns", [])
        if prop_columns:
            from hypertopos.builder._stats import SIGMA_EPS_PROP

            prop_start = len(pattern.relations) + len(pattern.event_dimensions)
            prop_end = prop_start + len(prop_columns)
            new_sigma[prop_start:prop_end] = np.maximum(
                new_sigma[prop_start:prop_end], SIGMA_EPS_PROP
            )

        new_deltas = calibrated_deltas(
            shape_vectors.astype(np.float32),
            mu=new_mu,
            sigma_diag=new_sigma,
            cholesky_inv=new_cholesky_inv,
            dimension_weights=pattern.dimension_weights,
        ).astype(np.float32)
        new_delta_norms = np.sqrt(
            np.einsum("ij,ij->i", new_deltas, new_deltas),
        ).astype(np.float32)
        D = shape_vectors.shape[1]
        theta_scalar = float(np.percentile(new_delta_norms[fit_mask], pct))
        component = theta_scalar / np.sqrt(D) if D > 0 else 0.0
        new_theta = np.full(D, component, dtype=np.float32)

        new_theta_norm = float(np.linalg.norm(new_theta))
        # Use >= (boundary inclusive) to match builder.py and geometry.py is_anomaly semantics.
        # Entities at exactly delta_norm == theta_norm are anomalous (on the boundary).
        is_anomaly = (new_theta_norm > 0.0) & (new_delta_norms >= new_theta_norm)

        # Same per-dim rule as the build: P99 of |delta| over the fitted entities.
        n_anom_dims_thresholds = per_dim_anomaly_thresholds(new_deltas[fit_mask])

        # 4. Compute delta_rank_pct
        sorted_norms = np.sort(new_delta_norms)
        ranks = np.searchsorted(sorted_norms, new_delta_norms, side="left")
        delta_rank_pct = (ranks / max(n, 1) * 100).astype(np.float32)

        # 5. The calibration-dependent columns, built from numpy buffers in the
        # row order of step 1's read. Every other column is left as stored.
        d = new_deltas.shape[1]
        # Columns derived from the old calibration are recomputed from the new
        # deltas; anomaly_confidence (bootstrap stability of the old fit) and
        # delta_norm_signed (projection on a label-aware direction that is not
        # stored with the pattern) cannot be, so they become null and readers
        # fall back to delta_norm.
        conformal_p = ((n - ranks) / max(n, 1)).astype(np.float32)
        dim_kinds = getattr(pattern, "dimension_kinds", None)
        if dim_kinds is not None:
            from hypertopos.builder._bregman import bregman_norms

            bregman = bregman_norms(
                shape_vectors.astype(np.float32),
                new_mu,
                new_sigma,
                dim_kinds,
                weights=pattern.dimension_weights,
            ).astype(np.float32)
        else:
            bregman = np.zeros(n, dtype=np.float32)
        values: dict[str, Any] = {
            "delta": pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(new_deltas, dtype=np.float32).ravel()), d
            ),
            "delta_norm": pa.array(new_delta_norms.astype(np.float32)),
            "is_anomaly": pa.array(is_anomaly.astype(np.bool_)),
            "delta_rank_pct": pa.array(delta_rank_pct.astype(np.float32)),
            "conformal_p": pa.array(conformal_p),
            "bregman_divergence": pa.array(bregman),
            "n_anomalous_dims": pa.array(
                compute_per_dim_anomaly_count(new_deltas, thresholds=n_anom_dims_thresholds)
            ),
            "anomaly_confidence": pa.nulls(n, type=pa.float32()),
            "delta_norm_signed": pa.nulls(n, type=pa.float32()),
        }
        for col_name in geo_columns:
            if col_name.startswith("delta_dim_") and int(col_name.split("_")[-1]) < d:
                values[col_name] = pa.array(
                    np.ascontiguousarray(new_deltas[:, int(col_name.split("_")[-1])])
                )
        values = {name: column for name, column in values.items() if name in geo_columns}

        # 6. Replace those columns in place in one commit (native MVCC, new
        # internal Lance version) and rebuild only their indexes; the key and
        # relation columns and their indexes are not rewritten. The values are
        # computed from the pinned geometry: refuse when another writer
        # committed after this session pinned its snapshot.
        pinned = self._reader.pinned_geometry_version(pattern_id)
        latest = self._reader.latest_lance_versions([pattern_id]).get(pattern_id)
        if pinned is not None and pinned != latest:
            raise GDSStorageError(
                f"geometry of '{pattern_id}' changed after this session opened; "
                "call refresh() and recalibrate again"
            )
        on_branch = self._reader.branch
        # Lance commits through the branch head on a branch; main is not touched.
        self._writer.replace_geometry_columns(geo_path, on_branch, row_ids, values)
        meta_writer = GDSWriter(str(meta_root)) if on_branch else self._writer

        # 7. Record the fit as the next calibration epoch (same schema, so the
        # previous epoch + 1), then point sphere.json at it. Fields computed
        # from the entity data rather than the fit (percentiles, normality,
        # edge_max, segment stats) carry over; theta sensitivity is recomputed.
        sphere_data = json.loads(sphere_path.read_text())
        pat_dict = sphere_data["patterns"][pattern_id]
        now = datetime.now(UTC)
        prev_fit = self._reader.read_calibration_fit(pattern_id)
        new_epoch = int(pat_dict.get("calibration_epoch", prev_fit.calibration_epoch)) + 1
        fit = dataclasses.replace(
            prev_fit,
            calibration_epoch=new_epoch,
            mu=np.asarray(new_mu, dtype=np.float32),
            sigma_diag=np.asarray(new_sigma, dtype=np.float32),
            theta=new_theta,
            population_size=n,
            last_calibrated_at=now,
            theta_sensitivity=compute_theta_sensitivity_from_sorted(
                np.sort(new_delta_norms[fit_mask])
            ),
        )
        last_k = self._reader.read_calibration_history_policy()["last_k"]
        if not history_dir(meta_root, pattern_id).exists():
            # The previous fit exists only inline in sphere.json; record it
            # first so it stays readable once the history directory exists.
            write_calibration_history_epoch(meta_root, prev_fit, last_k=last_k)
        write_calibration_history_epoch(meta_root, fit, last_k=last_k)

        pat_dict["mu"] = new_mu.tolist()
        pat_dict["sigma_diag"] = new_sigma.tolist()
        pat_dict["theta"] = new_theta.tolist()
        pat_dict["n_anomalous_dims_thresholds"] = n_anom_dims_thresholds.tolist()
        if new_cholesky_inv is not None:
            pat_dict["cholesky_inv"] = new_cholesky_inv.tolist()
        else:
            pat_dict.pop("cholesky_inv", None)
        pat_dict["last_calibrated_at"] = now.isoformat()
        pat_dict["population_size"] = n
        pat_dict["calibration_epoch"] = new_epoch
        write_sphere_json(sphere_path.parent, sphere_data)

        # 8. Update geometry_stats cache
        meta_writer.write_geometry_stats(
            pattern_id,
            version,
            new_delta_norms,
            new_theta_norm,
        )

        # 9. Invalidate temporal centroid cache (stale after recalibration)
        centroids_path = (
            self._reader._base / "_gds_meta" / "temporal_centroids" / f"{pattern_id}.lance"
        )
        if not on_branch:
            self._writer.invalidate_temporal_centroids(pattern_id)
        elif centroids_path.exists():
            self._writer.invalidate_temporal_centroids(
                pattern_id, _ds=self._reader.dataset(centroids_path)
            )

        # 10. Reset tracker
        old_tracker = self._reader.read_calibration_tracker(pattern_id)
        old_drift = old_tracker.drift_pct if old_tracker else 0.0
        s_thresh = (
            soft_threshold
            if soft_threshold is not None
            else (old_tracker.soft_threshold if old_tracker else 0.05)
        )
        h_thresh = (
            hard_threshold
            if hard_threshold is not None
            else (old_tracker.hard_threshold if old_tracker else 0.20)
        )
        tracker = CalibrationTracker.from_stats(
            new_mu,
            new_sigma,
            new_theta,
            n=n,
            soft_threshold=s_thresh,
            hard_threshold=h_thresh,
        )
        meta_writer.write_calibration_tracker(pattern_id, tracker)

        mu_sha256 = (
            "sha256:" + hashlib.sha256(np.asarray(new_mu, dtype=np.float64).tobytes()).hexdigest()
        )
        result = {
            "pattern_id": pattern_id,
            "previous_drift_pct": round(old_drift, 4),
            "new_theta_norm": round(new_theta_norm, 4),
            "old_theta_norm": round(old_theta_norm, 4),
            "records_recalibrated": n,
            "n_excluded": len(excluded),
            "branch": on_branch,
            "mu_sha256": mu_sha256,
        }
        if on_branch:
            # Recorded before the session re-pins, so a refresh that fails
            # under a concurrent writer cannot leave the commit unrecorded.
            after = self._reader._branch_versions()
            _lance_branches.append_recipe(
                self._reader._base,
                on_branch,
                {
                    "op": "recalibrate",
                    "pattern_id": pattern_id,
                    "exclude_keys": excluded,
                    "soft_threshold": soft_threshold,
                    "hard_threshold": hard_threshold,
                    "result": {
                        "theta_norm": new_theta_norm,
                        "population_size": n,
                        "n_excluded": len(excluded),
                        "mu_sha256": mu_sha256,
                    },
                    "lance_version_after": {
                        key: pin[1]
                        for key in (pattern_id, f"centroids:{pattern_id}")
                        if isinstance(pin := after.get(key), tuple)
                    },
                },
            )
        # The session reads its own write.
        self.refresh()
        return result

    def close(self, purge_temporal: bool = False) -> None:
        self._manifest.status = "expired"
        if purge_temporal:
            self._writer.purge_agent_temporal(self._manifest.agent_id)

    def __enter__(self) -> HyperSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
