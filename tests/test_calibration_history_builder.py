"""Integration + helper tests for the builder side of multi-epoch retention.

Builder integration tests build real spheres from synthetic data and assert
sphere.json + calibration_history/ on disk."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_compute_schema_hash_for_pattern_def_matches_node_round_trip():
    """The hash from a builder's Pattern definition must equal the hash from
    the sphere.json pattern node that builder writes."""
    from hypertopos.builder.builder import _compute_schema_hash_for_pattern_def
    from hypertopos.storage.calibration_history import _compute_schema_hash_from_pattern_node

    class _Rel:
        def __init__(self, line_id, event_columns):
            self.line_id = line_id
            self.event_columns = event_columns

    class _Pat:
        relations = [_Rel("tx", ["amount", "ts"]), _Rel("acct", [])]
        event_dimensions = ["amount_zscore"]
        prop_columns = ["country"]
        dimension_kinds = ["gaussian", "gaussian", "bernoulli"]

    pat = _Pat()
    h_def = _compute_schema_hash_for_pattern_def(pat)

    node = {
        "relations": [
            {"line_id": "tx", "event_columns": ["amount", "ts"]},
            {"line_id": "acct", "event_columns": []},
        ],
        "event_dimensions": ["amount_zscore"],
        "prop_columns": ["country"],
        "dimension_kinds": ["gaussian", "gaussian", "bernoulli"],
    }
    h_node = _compute_schema_hash_from_pattern_node(node)
    assert h_def == h_node


# ---------------------------------------------------------------------------
# tiny_sphere_factory — builds a minimal 1-pattern event sphere into a path.
# Calling it twice on the same path exercises the increment branch.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_sphere_factory():
    """Returns a callable that builds a tiny 1-pattern sphere into a given path.

    The factory can be called multiple times with the same path to trigger
    rebuild (exercises the calibration epoch increment branch).

    On subsequent calls the factory removes points/ and geometry/ so the builder
    can write fresh Lance datasets while preserving _gds_meta/ (sphere.json +
    calibration_history/).  This replicates how an operator would re-run
    calibration on a sphere that has already been built once.

    Returns the pattern_id ``"gl_entry_pattern"`` on every call.
    """
    import shutil

    from hypertopos.builder import GDSBuilder, RelationSpec

    def _build(sphere_path: Path) -> str:
        sphere_path.mkdir(parents=True, exist_ok=True)

        # Remove Lance-backed directories from a prior build so the builder
        # can create them fresh.  _gds_meta/ is intentionally kept so that
        # _write_calibration_epoch_for_pattern can read the prior sphere.json.
        for subdir in ("points", "geometry", "temporal"):
            target = sphere_path / subdir
            if target.exists():
                shutil.rmtree(target)

        b = GDSBuilder("test", str(sphere_path))
        b.add_line(
            "customers",
            [
                {"cust_id": "C-1", "name": "Alpha"},
                {"cust_id": "C-2", "name": "Beta"},
                {"cust_id": "C-3", "name": "Gamma"},
                {"cust_id": "C-4", "name": "Delta"},
                {"cust_id": "C-5", "name": "Epsilon"},
            ],
            key_col="cust_id",
            source_id="test",
        )
        b.add_line(
            "company_codes",
            [
                {"cc_id": "CC-01", "name": "Germany"},
                {"cc_id": "CC-02", "name": "USA"},
            ],
            key_col="cc_id",
            source_id="test",
        )
        b.add_line(
            "gl_entries",
            [
                {"doc": "D-001", "cust_id": "C-1", "cc_id": "CC-01"},
                {"doc": "D-002", "cust_id": "C-1", "cc_id": "CC-01"},
                {"doc": "D-003", "cust_id": "C-2", "cc_id": "CC-02"},
                {"doc": "D-004", "cust_id": "C-3", "cc_id": "CC-01"},
                {"doc": "D-005", "cust_id": "C-4", "cc_id": "CC-01"},
                {"doc": "D-006", "cust_id": None, "cc_id": "CC-01"},
                {"doc": "D-007", "cust_id": None, "cc_id": "CC-02"},
                {"doc": "D-008", "cust_id": None, "cc_id": None},
            ],
            key_col="doc",
            source_id="test",
            role="event",
        )
        b.add_pattern(
            "gl_entry_pattern",
            pattern_type="event",
            entity_line="gl_entries",
            relations=[
                RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
                RelationSpec("company_codes", fk_col="cc_id", direction="in", required=True),
            ],
            anomaly_percentile=60.0,
        )
        b.build()
        return "gl_entry_pattern"

    return _build


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_build_once_writes_v1(tiny_sphere_factory, tmp_path):
    """First builder run on a fresh dir writes v=1.json with schema_hash and
    sphere.json with format_version=3.0 + calibration_epoch=1."""
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)

    meta = json.loads((sphere_path / "_gds_meta" / "sphere.json").read_text())
    assert meta["format_version"] == "3.0"
    assert meta["calibration_history_policy"] == {"last_k": 5}
    assert meta["patterns"][pid]["calibration_epoch"] == 1
    assert len(meta["patterns"][pid]["schema_hash"]) == 64

    epoch_path = sphere_path / "_gds_meta" / "calibration_history" / pid / "v=1.json"
    assert epoch_path.exists()
    blob = json.loads(epoch_path.read_text())
    assert blob["calibration_epoch"] == 1
    assert blob["schema_hash"] == meta["patterns"][pid]["schema_hash"]

    reader = GDSReader(sphere_path)
    fit = reader.read_calibration_fit(pid)
    assert fit.calibration_epoch == 1


def test_build_twice_increments_epoch(tiny_sphere_factory, tmp_path):
    """Second build on the same path increments calibration_epoch to 2 and
    writes v=2.json; both epochs are readable by the reader."""
    from hypertopos.storage.reader import GDSReader
    import numpy as np

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)
    pid2 = tiny_sphere_factory(sphere_path)
    assert pid == pid2

    meta = json.loads((sphere_path / "_gds_meta" / "sphere.json").read_text())
    assert meta["patterns"][pid]["calibration_epoch"] == 2
    assert (sphere_path / "_gds_meta" / "calibration_history" / pid / "v=2.json").exists()

    reader = GDSReader(sphere_path)
    fit_v1 = reader.read_calibration_fit(pid, version=1)
    fit_v2 = reader.read_calibration_fit(pid, version=2)
    np.testing.assert_array_equal(fit_v1.mu, fit_v2.mu)
    assert fit_v1.schema_hash == fit_v2.schema_hash


def test_six_builds_with_last_k_5_keeps_v2_through_v6(tiny_sphere_factory, tmp_path):
    """Six builds with default last_k=5 retain only the five most recent epochs
    (v=2 through v=6), discarding v=1."""
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    for _ in range(6):
        pid = tiny_sphere_factory(sphere_path)

    reader = GDSReader(sphere_path)
    assert reader.list_calibration_versions(pid) == [2, 3, 4, 5, 6]

    meta = json.loads((sphere_path / "_gds_meta" / "sphere.json").read_text())
    assert meta["patterns"][pid]["calibration_epoch"] == 6


def test_last_k_one_effective_replace(tiny_sphere_factory, tmp_path):
    """When calibration_history_policy.last_k=1 is written into sphere.json
    before a rebuild, that rebuild's GC trims to a single epoch on disk.

    Note: the builder always re-emits last_k=5 in the sphere.json it writes,
    so this test validates only the *first* rebuild after the manual last_k=1
    injection.  Only v=2.json should survive that build; subsequent builds
    revert to last_k=5 retention.
    """
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)          # epoch 1, last_k=5

    sp = sphere_path / "_gds_meta" / "sphere.json"
    meta = json.loads(sp.read_text())
    meta["calibration_history_policy"] = {"last_k": 1}
    sp.write_text(json.dumps(meta))

    tiny_sphere_factory(sphere_path)                 # epoch 2, reads last_k=1 → trims to v=2 only

    reader = GDSReader(sphere_path)
    versions_after_trim = reader.list_calibration_versions(pid)
    assert versions_after_trim == [2], (
        f"Expected only [2] after last_k=1 trim, got {versions_after_trim}"
    )


# ---------------------------------------------------------------------------
# Additional fixtures for edge-case tests
# ---------------------------------------------------------------------------

def _build_tiny_sphere(
    sphere_path: Path,
    *,
    with_event_dim: bool = False,
    patterns: list[str] | None = None,
) -> list[str]:
    """Shared sphere-construction logic used by multiple fixtures.

    Parameters
    ----------
    sphere_path:
        Target directory for the sphere.
    with_event_dim:
        When ``True``, adds a numeric ``amount`` column to gl_entries and
        registers it as an event dimension on ``gl_entry_pattern``.  This
        changes ``event_dimensions`` (and therefore ``schema_hash``) relative
        to the default build, enabling schema-drift tests without touching
        RelationSpec edge_max (which requires numeric FK columns).
    patterns:
        List of pattern ids to build.  Supported: ``"gl_entry_pattern"``
        (event, always) and ``"customer_pattern"`` (anchor, optional).
        Defaults to ``["gl_entry_pattern"]``.

    Returns
    -------
    list[str]
        Pattern ids that were built (in the same order as *patterns*).
    """
    import shutil

    from hypertopos.builder import GDSBuilder, RelationSpec

    if patterns is None:
        patterns = ["gl_entry_pattern"]

    sphere_path.mkdir(parents=True, exist_ok=True)
    for subdir in ("points", "geometry", "temporal"):
        target = sphere_path / subdir
        if target.exists():
            shutil.rmtree(target)

    b = GDSBuilder("test", str(sphere_path))
    b.add_line(
        "customers",
        [
            {"cust_id": "C-1", "name": "Alpha"},
            {"cust_id": "C-2", "name": "Beta"},
            {"cust_id": "C-3", "name": "Gamma"},
            {"cust_id": "C-4", "name": "Delta"},
            {"cust_id": "C-5", "name": "Epsilon"},
        ],
        key_col="cust_id",
        source_id="test",
    )
    b.add_line(
        "company_codes",
        [
            {"cc_id": "CC-01", "name": "Germany"},
            {"cc_id": "CC-02", "name": "USA"},
        ],
        key_col="cc_id",
        source_id="test",
    )

    # gl_entries always has the amount column so the schema is consistent;
    # whether it's registered as an event_dimension is controlled separately.
    b.add_line(
        "gl_entries",
        [
            {"doc": "D-001", "cust_id": "C-1", "cc_id": "CC-01", "amount": 100.0},
            {"doc": "D-002", "cust_id": "C-1", "cc_id": "CC-01", "amount": 200.0},
            {"doc": "D-003", "cust_id": "C-2", "cc_id": "CC-02", "amount": 150.0},
            {"doc": "D-004", "cust_id": "C-3", "cc_id": "CC-01", "amount": 300.0},
            {"doc": "D-005", "cust_id": "C-4", "cc_id": "CC-01", "amount": 250.0},
            {"doc": "D-006", "cust_id": None, "cc_id": "CC-01", "amount": 50.0},
            {"doc": "D-007", "cust_id": None, "cc_id": "CC-02", "amount": 75.0},
            {"doc": "D-008", "cust_id": None, "cc_id": None,    "amount": 10.0},
        ],
        key_col="doc",
        source_id="test",
        role="event",
    )

    built: list[str] = []

    if "gl_entry_pattern" in patterns:
        b.add_pattern(
            "gl_entry_pattern",
            pattern_type="event",
            entity_line="gl_entries",
            relations=[
                RelationSpec(
                    "customers",
                    fk_col="cust_id",
                    direction="in",
                    required=False,
                ),
                RelationSpec(
                    "company_codes",
                    fk_col="cc_id",
                    direction="in",
                    required=True,
                ),
            ],
            anomaly_percentile=60.0,
        )
        if with_event_dim:
            b.add_event_dimension("gl_entry_pattern", "amount")
        built.append("gl_entry_pattern")

    if "customer_pattern" in patterns:
        # Anchor pattern — uses tracked_properties as its single dimension source
        # (string category → bernoulli dimension).
        b.add_pattern(
            "customer_pattern",
            pattern_type="anchor",
            entity_line="customers",
            relations=[],
            tracked_properties=["name"],
            anomaly_percentile=60.0,
        )
        built.append("customer_pattern")

    b.build()
    return built


@pytest.fixture
def tiny_sphere_factory_with_dim_kind():
    """Variant of tiny_sphere_factory that lets the caller vary the schema
    by including or excluding an event dimension, changing schema_hash.

    Usage::

        pid = factory(sphere_path, dim_kind="bernoulli")   # 2 FK dims only
        pid = factory(sphere_path, dim_kind="gaussian")    # adds amount dim

    The ``dim_kind`` parameter is used as a semantic label:
    ``"bernoulli"`` → no event dimension (FK-only schema),
    ``"gaussian"``  → adds the ``amount`` event dimension (gaussian kind).

    Returns the pattern_id ``"gl_entry_pattern"`` on every call.
    """
    def _build(sphere_path: Path, dim_kind: str = "bernoulli") -> str:
        # bernoulli → FK-only schema (2 dims, both bernoulli)
        # gaussian  → adds a continuous event dimension → different schema_hash
        with_event_dim = dim_kind != "bernoulli"
        pids = _build_tiny_sphere(sphere_path, with_event_dim=with_event_dim)
        return pids[0]

    return _build


@pytest.fixture
def tiny_two_pattern_factory():
    """Builds a sphere with one or two patterns.

    ``include_b=False``  → only ``"gl_entry_pattern"`` (event)
    ``include_b=True``   → adds ``"customer_pattern"`` (anchor)

    Returns ``(pid_a, pid_b)`` where ``pid_b`` is ``None`` when
    ``include_b=False``.
    """
    def _build(sphere_path: Path, include_b: bool) -> tuple[str, str | None]:
        patterns = ["gl_entry_pattern"]
        if include_b:
            patterns.append("customer_pattern")
        pids = _build_tiny_sphere(sphere_path, patterns=patterns)
        pid_a = pids[0]
        pid_b = pids[1] if include_b else None
        return pid_a, pid_b

    return _build


# ---------------------------------------------------------------------------
# Edge-case integration tests
# ---------------------------------------------------------------------------

def test_schema_change_resets_to_v1(tiny_sphere_factory_with_dim_kind, tmp_path):
    """Changing dimension_kinds triggers schema_hash drift → wipes history,
    resets to v=1."""
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    tiny_sphere_factory_with_dim_kind(sphere_path, dim_kind="bernoulli")
    pid = tiny_sphere_factory_with_dim_kind(sphere_path, dim_kind="bernoulli")
    # History now has v=1 and v=2.

    h_before = json.loads(
        (sphere_path / "_gds_meta" / "sphere.json").read_text()
    )["patterns"][pid]["schema_hash"]

    # Re-build with gaussian event dim — schema_hash changes → reset
    pid = tiny_sphere_factory_with_dim_kind(sphere_path, dim_kind="gaussian")

    meta = json.loads((sphere_path / "_gds_meta" / "sphere.json").read_text())
    assert meta["patterns"][pid]["calibration_epoch"] == 1
    assert meta["patterns"][pid]["schema_hash"] != h_before

    reader = GDSReader(sphere_path)
    assert reader.list_calibration_versions(pid) == [1]


def test_pattern_added_gets_v1_independently(tiny_two_pattern_factory, tmp_path):
    """When a builder run adds a brand-new pattern, that pattern starts at v=1
    while other patterns continue their counter."""
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    tiny_two_pattern_factory(sphere_path, include_b=False)
    tiny_two_pattern_factory(sphere_path, include_b=False)
    # gl_entry_pattern has v=1, v=2

    pid_a, pid_b = tiny_two_pattern_factory(sphere_path, include_b=True)

    reader = GDSReader(sphere_path)
    assert reader.list_calibration_versions(pid_a) == [1, 2, 3]
    assert reader.list_calibration_versions(pid_b) == [1]


def test_pattern_removed_leaves_orphan_dir(tiny_two_pattern_factory, tmp_path):
    """When a pattern is dropped from the builder definition, its history dir
    is left untouched on disk."""
    sphere_path = tmp_path / "sphere"
    pid_a, pid_b = tiny_two_pattern_factory(sphere_path, include_b=True)

    # Second run omits customer_pattern — orphan dir must survive
    tiny_two_pattern_factory(sphere_path, include_b=False)

    orphan_dir = sphere_path / "_gds_meta" / "calibration_history" / pid_b
    assert orphan_dir.exists()
    assert (orphan_dir / "v=1.json").exists()


def test_legacy_sphere_first_build_writes_3_0(tiny_sphere_factory, tmp_path):
    """A legacy (pre-3.0) sphere.json on disk is rewritten as 3.0 by the
    first builder run — the builder always emits 3.0 regardless of prior
    format_version, since the only supported reader path requires 3.0."""
    sphere_path = tmp_path / "sphere"
    sphere_path.mkdir(parents=True)

    # Build a valid sphere first, then mark sphere.json as legacy.
    pid = tiny_sphere_factory(sphere_path)
    sphere_path_meta = sphere_path / "_gds_meta" / "sphere.json"
    meta = json.loads(sphere_path_meta.read_text())
    meta["format_version"] = "2.3"
    meta.pop("calibration_history_policy", None)
    for p_node in meta["patterns"].values():
        p_node.pop("calibration_epoch", None)
        p_node.pop("schema_hash", None)
    sphere_path_meta.write_text(json.dumps(meta, indent=2))

    # Wipe history dir to fully simulate a legacy sphere.
    history_root = sphere_path / "_gds_meta" / "calibration_history"
    if history_root.exists():
        import shutil
        shutil.rmtree(history_root)

    # Re-build — fresh 3.0 metadata.
    pid = tiny_sphere_factory(sphere_path)

    meta_after = json.loads(sphere_path_meta.read_text())
    assert meta_after["format_version"] == "3.0"
    assert meta_after["calibration_history_policy"] == {"last_k": 5}
    assert meta_after["patterns"][pid]["calibration_epoch"] == 1
    assert "schema_hash" in meta_after["patterns"][pid]
    assert (
        sphere_path / "_gds_meta" / "calibration_history" / pid / "v=1.json"
    ).exists()


# ---------------------------------------------------------------------------
# P13 — Welford sidecar isolation
# ---------------------------------------------------------------------------

def test_welford_sidecar_byte_identical_after_history_write(tiny_sphere_factory, tmp_path):
    """Writing a new history epoch must NOT touch _gds_meta/calibration/{pid}.json."""
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import write_calibration_history_epoch
    import shutil
    from datetime import datetime, timezone
    import numpy as np

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)
    welford_path = sphere_path / "_gds_meta" / "calibration" / f"{pid}.json"
    if not welford_path.exists():
        pytest.skip("Welford sidecar not written by tiny_sphere_factory; revisit fixture")

    welford_baseline_bytes = welford_path.read_bytes()

    work = tmp_path / "work"
    shutil.copytree(sphere_path, work)
    fit = CalibrationFit(
        pattern_id=pid,
        calibration_epoch=99,
        schema_version=1,
        schema_hash="z" * 64,
        mu=np.array([0.0], dtype=np.float32),
        sigma_diag=np.array([1.0], dtype=np.float32),
        theta=np.array([3.0], dtype=np.float32),
        population_size=1,
        dimension_weights=None,
        dimension_kinds=None,
        dim_percentiles=None,
        group_stats=None,
        gmm_components=None,
        edge_max=None,
        computed_at=datetime.now(timezone.utc),
        last_calibrated_at=datetime.now(timezone.utc),
    )
    write_calibration_history_epoch(work, fit, last_k=5)

    work_welford = work / "_gds_meta" / "calibration" / f"{pid}.json"
    assert work_welford.read_bytes() == welford_baseline_bytes


def test_welford_sidecar_byte_identical_after_gc(tiny_sphere_factory, tmp_path):
    """GC (trimming oldest history files) must NOT touch the Welford sidecar."""
    from hypertopos.model.sphere import CalibrationFit
    from hypertopos.storage.calibration_history import write_calibration_history_epoch
    import shutil
    from datetime import datetime, timezone
    import numpy as np

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)
    welford_path = sphere_path / "_gds_meta" / "calibration" / f"{pid}.json"
    if not welford_path.exists():
        pytest.skip("Welford sidecar not written by tiny_sphere_factory")

    work = tmp_path / "work"
    shutil.copytree(sphere_path, work)
    welford_baseline_bytes = (work / "_gds_meta" / "calibration" / f"{pid}.json").read_bytes()

    for n in range(2, 9):
        fit = CalibrationFit(
            pattern_id=pid, calibration_epoch=n, schema_version=1, schema_hash="z" * 64,
            mu=np.array([0.0], dtype=np.float32), sigma_diag=np.array([1.0], dtype=np.float32),
            theta=np.array([3.0], dtype=np.float32), population_size=1,
            dimension_weights=None, dimension_kinds=None, dim_percentiles=None,
            group_stats=None, gmm_components=None, edge_max=None,
            computed_at=datetime.now(timezone.utc), last_calibrated_at=datetime.now(timezone.utc),
        )
        write_calibration_history_epoch(work, fit, last_k=5)

    assert (work / "_gds_meta" / "calibration" / f"{pid}.json").read_bytes() == welford_baseline_bytes


def test_welford_sidecar_byte_identical_after_reset(tiny_sphere_factory, tmp_path):
    """reset_calibration_history must NOT touch the Welford sidecar."""
    from hypertopos.storage.calibration_history import reset_calibration_history
    import shutil

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)
    welford_path = sphere_path / "_gds_meta" / "calibration" / f"{pid}.json"
    if not welford_path.exists():
        pytest.skip("Welford sidecar not written by tiny_sphere_factory")

    work = tmp_path / "work"
    shutil.copytree(sphere_path, work)
    welford_baseline_bytes = (work / "_gds_meta" / "calibration" / f"{pid}.json").read_bytes()

    reset_calibration_history(work, pid)

    assert (work / "_gds_meta" / "calibration" / f"{pid}.json").read_bytes() == welford_baseline_bytes


# ---------------------------------------------------------------------------
# P14 — dim_percentiles + gmm_components round-trip on built sphere
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_sphere_with_percentiles_factory():
    """Builds a tiny sphere whose entity line has a float column, so the
    builder's _compute_dim_percentiles produces a non-empty dict."""
    def _build(sphere_path: Path) -> str:
        # _build_tiny_sphere always includes `amount` (float) on gl_entries.
        # _compute_dim_percentiles scans all float columns in the entity line
        # table, so dim_percentiles is produced automatically.
        pids = _build_tiny_sphere(sphere_path, with_event_dim=True)
        return pids[0]

    return _build


@pytest.fixture
def tiny_sphere_with_gmm_factory():
    """Builds a tiny sphere with gmm_n_components=2, so the builder fits GMM
    and writes gmm_components into v=1.json."""
    import shutil

    from hypertopos.builder import GDSBuilder, RelationSpec

    def _build(sphere_path: Path) -> str:
        sphere_path.mkdir(parents=True, exist_ok=True)
        for subdir in ("points", "geometry", "temporal"):
            target = sphere_path / subdir
            if target.exists():
                shutil.rmtree(target)

        b = GDSBuilder("test", str(sphere_path))
        b.add_line(
            "customers",
            [
                {"cust_id": "C-1", "name": "Alpha"},
                {"cust_id": "C-2", "name": "Beta"},
                {"cust_id": "C-3", "name": "Gamma"},
                {"cust_id": "C-4", "name": "Delta"},
                {"cust_id": "C-5", "name": "Epsilon"},
            ],
            key_col="cust_id",
            source_id="test",
        )
        b.add_line(
            "company_codes",
            [
                {"cc_id": "CC-01", "name": "Germany"},
                {"cc_id": "CC-02", "name": "USA"},
            ],
            key_col="cc_id",
            source_id="test",
        )
        b.add_line(
            "gl_entries",
            [
                {"doc": "D-001", "cust_id": "C-1", "cc_id": "CC-01", "amount": 100.0},
                {"doc": "D-002", "cust_id": "C-1", "cc_id": "CC-01", "amount": 200.0},
                {"doc": "D-003", "cust_id": "C-2", "cc_id": "CC-02", "amount": 150.0},
                {"doc": "D-004", "cust_id": "C-3", "cc_id": "CC-01", "amount": 300.0},
                {"doc": "D-005", "cust_id": "C-4", "cc_id": "CC-01", "amount": 250.0},
                {"doc": "D-006", "cust_id": None, "cc_id": "CC-01", "amount": 50.0},
                {"doc": "D-007", "cust_id": None, "cc_id": "CC-02", "amount": 75.0},
                {"doc": "D-008", "cust_id": None, "cc_id": None,    "amount": 10.0},
            ],
            key_col="doc",
            source_id="test",
            role="event",
        )
        b.add_pattern(
            "gl_entry_pattern",
            pattern_type="event",
            entity_line="gl_entries",
            relations=[
                RelationSpec("customers", fk_col="cust_id", direction="in", required=False),
                RelationSpec("company_codes", fk_col="cc_id", direction="in", required=True),
            ],
            anomaly_percentile=60.0,
            gmm_n_components=2,
        )
        b.build()
        return "gl_entry_pattern"

    return _build


def test_dim_percentiles_round_trip_on_built_sphere(tiny_sphere_with_percentiles_factory, tmp_path):
    """A pattern whose entity line has float columns must produce dim_percentiles
    that survive the v=N.json round-trip."""
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_with_percentiles_factory(sphere_path)

    reader = GDSReader(sphere_path)
    fit = reader.read_calibration_fit(pid)
    assert fit.dim_percentiles is not None, (
        "Builder did not produce dim_percentiles for a pattern with float entity columns"
    )
    assert isinstance(fit.dim_percentiles, dict)
    assert len(fit.dim_percentiles) > 0


def test_gmm_components_round_trip_on_built_sphere(tiny_sphere_with_gmm_factory, tmp_path):
    """A pattern built with gmm_n_components=2 must produce gmm_components that
    survive the v=N.json round-trip."""
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_with_gmm_factory(sphere_path)

    reader = GDSReader(sphere_path)
    fit = reader.read_calibration_fit(pid)
    assert fit.gmm_components is not None, (
        "Builder did not produce gmm_components for a pattern with gmm_n_components=2"
    )
    assert isinstance(fit.gmm_components, list)
    assert len(fit.gmm_components) > 0


def test_sphere_overview_surfaces_theta_sensitivity_summary(tiny_sphere_factory, tmp_path):
    """sphere_overview entry on a built sphere must carry a populated
    theta_sensitivity_summary block: stable_band_from/to/length, n_cliffs,
    theta_at_p95. Validates the T4 sphere_overview integration."""
    from hypertopos.sphere import HyperSphere

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)

    sphere = HyperSphere.open(str(sphere_path))
    session = sphere.session(agent_id="overview_ts_test")
    try:
        nav = session.navigator()
        entries = nav.sphere_overview()
        match = next((e for e in entries if e["pattern_id"] == pid), None)
        assert match is not None
        assert "theta_sensitivity_summary" in match
        ts = match["theta_sensitivity_summary"]
        assert "stable_band_from" in ts
        assert "stable_band_to" in ts
        assert "stable_band_length" in ts
        assert "n_cliffs" in ts
        assert "theta_at_p95" in ts
        assert isinstance(ts["stable_band_length"], int)
        assert isinstance(ts["n_cliffs"], int)
        assert ts["theta_at_p95"] is None or isinstance(
            ts["theta_at_p95"], float,
        )
    finally:
        session.close()


def test_theta_sensitivity_navigator_method_on_built_sphere(tiny_sphere_factory, tmp_path):
    """Built sphere → navigator.theta_sensitivity returns a populated
    ThetaSensitivityReport with derived stable_band + cliffs. End-to-end
    integration test for the T3 surface."""
    from hypertopos.model.sphere import ThetaSensitivityReport
    from hypertopos.sphere import HyperSphere

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)

    sphere = HyperSphere.open(str(sphere_path))
    session = sphere.session(agent_id="theta_sens_test")
    try:
        nav = session.navigator()
        report = nav.theta_sensitivity(pid)
        assert isinstance(report, ThetaSensitivityReport)
        assert report.pattern_id == pid
        assert report.calibration_epoch == 1
        assert set(report.theta_sensitivity.keys()) == {f"p{p}" for p in range(90, 100)}
        assert isinstance(report.stable_band, dict)
        assert "from" in report.stable_band
        assert "to" in report.stable_band
        assert "length" in report.stable_band
        assert isinstance(report.cliffs, list)
        assert report.n_cliffs == len(report.cliffs)
        assert report.stable_band_length == report.stable_band["length"]
    finally:
        session.close()


def test_theta_sensitivity_navigator_raises_on_missing_pattern(tiny_sphere_factory, tmp_path):
    """Pattern that doesn't exist on disk → ValueError ('no calibration
    epochs on disk')."""
    from hypertopos.sphere import HyperSphere

    sphere_path = tmp_path / "sphere"
    tiny_sphere_factory(sphere_path)

    sphere = HyperSphere.open(str(sphere_path))
    session = sphere.session(agent_id="theta_sens_missing")
    try:
        nav = session.navigator()
        with pytest.raises(ValueError, match="no calibration epochs"):
            nav.theta_sensitivity("nonexistent_pattern")
    finally:
        session.close()


def test_theta_sensitivity_round_trip_on_built_sphere(tiny_sphere_factory, tmp_path):
    """Builder must populate `theta_sensitivity` on every CalibrationFit
    (cheap-path glued onto the existing `sorted_norms` for delta_rank_pcts).
    Field must survive the v=N.json round-trip with all five expected stat
    fields per percentile."""
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)

    reader = GDSReader(sphere_path)
    fit = reader.read_calibration_fit(pid)
    assert fit.theta_sensitivity is not None, (
        "Builder did not populate theta_sensitivity"
    )
    assert isinstance(fit.theta_sensitivity, dict)
    # Default percentile sweep p90..p99
    assert set(fit.theta_sensitivity.keys()) == {f"p{p}" for p in range(90, 100)}
    expected_fields = {
        "theta_mean", "theta_std",
        "anomaly_count_mean", "anomaly_count_std",
        "anomaly_rate",
    }
    for p_key, stats in fit.theta_sensitivity.items():
        assert set(stats.keys()) == expected_fields, f"missing fields at {p_key}"
    # Cheap path → std fields are 0.0
    for p in range(90, 100):
        assert fit.theta_sensitivity[f"p{p}"]["theta_std"] == 0.0
        assert fit.theta_sensitivity[f"p{p}"]["anomaly_count_std"] == 0.0
    # anomaly_count is monotonically non-increasing in percentile
    counts = [
        fit.theta_sensitivity[f"p{p}"]["anomaly_count_mean"] for p in range(90, 100)
    ]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1]


def test_compare_calibrations_on_built_sphere(tiny_sphere_factory, tmp_path):
    """Build the same tiny sphere twice — drift between v=1 and v=2 should
    be near-zero (same data, only theta bootstrap noise) and the report
    fields should be sensible."""
    import json as _json

    from hypertopos.sphere import HyperSphere

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)
    pid_again = tiny_sphere_factory(sphere_path)
    assert pid == pid_again

    sphere = HyperSphere.open(str(sphere_path))
    session = sphere.session(agent_id="m2_test")
    try:
        nav = session.navigator()
        report = nav.compare_calibrations(pid, top_n=5, verbose=True)
        assert report.v_from == 1
        assert report.v_to == 2
        assert report.overall_drift_rms < 0.01
        assert report.per_dimension is not None
        assert len(report.per_dimension) >= 1
        meta = _json.loads((sphere_path / "_gds_meta" / "sphere.json").read_text())
        assert report.schema_hash == meta["patterns"][pid]["schema_hash"]
    finally:
        session.close()


def test_decompose_drift_on_built_sphere(tiny_sphere_factory, tmp_path):
    """Build the same tiny sphere twice — drift between v=1 and v=2 should
    be near-zero (same data, only theta bootstrap noise) and the report
    fields should be sensible. Skipped if the tiny sphere has no temporal
    history for the pattern (M3 requires anchor + temporal)."""
    import json as _json

    from hypertopos.sphere import HyperSphere
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)
    pid_again = tiny_sphere_factory(sphere_path)
    assert pid == pid_again

    reader = GDSReader(sphere_path)
    sphere_obj_check = reader.read_sphere()
    pattern = sphere_obj_check.patterns[pid]
    if pattern.pattern_type != "anchor":
        import pytest as _pt
        _pt.skip(f"tiny_sphere_factory pattern is {pattern.pattern_type}; M3 requires anchor")

    sphere_obj = HyperSphere.open(str(sphere_path))
    session = sphere_obj.session(agent_id="m3_test")
    try:
        nav = session.navigator()
        candidate_keys = ["C-1", "C-2", "C-3", "C-4", "C-5"]
        chosen = None
        for k in candidate_keys:
            tbl = reader.read_temporal(pid, k, agent_id="m3_test")
            if tbl.num_rows >= 2:
                chosen = k
                break
        if chosen is None:
            import pytest as _pt
            _pt.skip(
                "tiny_sphere_factory produced no entity with >= 2 temporal slices; "
                "M3 cannot smoke-test"
            )

        report = nav.decompose_drift(chosen, pid, top_n=5, verbose=True)
        assert report.v_from == 1
        assert report.v_to == 2
        assert report.intrinsic_displacement >= 0.0
        assert report.extrinsic_displacement >= 0.0
        assert 0.0 <= report.intrinsic_fraction <= 1.0
        assert report.per_dimension is not None
        meta = _json.loads((sphere_path / "_gds_meta" / "sphere.json").read_text())
        assert report.schema_hash == meta["patterns"][pid]["schema_hash"]
    finally:
        session.close()


def test_find_calibration_influencers_on_built_sphere(tiny_sphere_factory, tmp_path):
    """Build the tiny sphere once, run find_calibration_influencers(classify='all'),
    verify cell_counts sum to N, classifications consistent with delta_norm
    + total_impact ranking. Skipped if the tiny sphere has no anchor pattern
    (M4 requires anchor — population statistics undefined for event)."""
    from hypertopos.sphere import HyperSphere
    from hypertopos.storage.reader import GDSReader

    sphere_path = tmp_path / "sphere"
    pid = tiny_sphere_factory(sphere_path)

    reader = GDSReader(sphere_path)
    sphere_obj_check = reader.read_sphere()
    pattern = sphere_obj_check.patterns[pid]
    if pattern.pattern_type != "anchor":
        import pytest as _pt
        _pt.skip(f"tiny_sphere_factory pattern is {pattern.pattern_type}; M4 requires anchor")

    sphere_obj = HyperSphere.open(str(sphere_path))
    session = sphere_obj.session(agent_id="m4_test")
    try:
        nav = session.navigator()
        report = nav.find_calibration_influencers(
            pattern_id=pid, classify="all", top_n=5,
        )
        assert report.population_size >= 2
        assert sum(report.cell_counts.values()) == report.population_size
        assert len(report.entries) <= 5
        for entry in report.entries:
            assert entry.classification in {
                "hidden", "distorter", "standard_anomaly", "normal",
            }
            assert entry.total_impact >= 0.0
            assert len(entry.top_dim_contributions) >= 1
    finally:
        session.close()
