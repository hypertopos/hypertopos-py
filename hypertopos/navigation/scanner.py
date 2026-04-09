# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""PassiveScanner — batch multi-source anomaly screening."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from hypertopos.model.manifest import Manifest
    from hypertopos.model.sphere import Sphere
    from hypertopos.storage.reader import GDSReader

logger = logging.getLogger(__name__)

_COMPOSITE_SEP = "\u2192"  # → separator used by add_composite_line


@dataclass
class ScanSource:
    """A registered anomaly source for passive scanning."""

    name: str
    pattern_id: str
    key_type: Literal["direct", "sibling", "composite", "chain", "borderline", "points", "compound"]
    weight: float = 1.0
    filter_expr: str | None = None
    # -- borderline --
    rank_threshold: float | None = None
    # -- points / compound --
    line_id: str | None = None
    rules: dict[str, tuple[str, float]] | None = None
    rules_combine: Literal["AND", "OR"] = "AND"
    # -- compound --
    geometry_pattern_id: str | None = None
    geometry_key_type: Literal["direct", "composite", "chain"] | None = None
    chain_filter_expr: str | None = None  # Lance SQL filter on chain points table


@dataclass
class ScanSourceHit:
    """Per-source detail for one entity."""

    anomalous_count: int
    related_count: int
    max_delta_norm: float
    anomaly_intensity: float = 0.0


@dataclass
class ScanHit:
    """A single entity's scan result."""

    primary_key: str
    score: int
    weighted_score: float
    sources: dict[str, ScanSourceHit] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Complete batch scan result."""

    home_line_id: str
    total_entities: int
    total_flagged: int
    hits: list[ScanHit]
    sources_summary: dict[str, int]
    elapsed_ms: float


def _apply_simple_filter(table: "pa.Table", expr: str) -> "pa.Table":  # type: ignore[name-defined]
    """Apply a simple column comparison filter to a PyArrow table.

    Supports: "col >= val", "col > val", "col <= val", "col < val", "col = val".
    val may be numeric or boolean (true/false).
    """
    import pyarrow.compute as pc  # local import — scanner has no top-level pyarrow dep

    for op in (">=", "<=", ">", "<", "="):
        if op in expr:
            col, raw = [p.strip() for p in expr.split(op, 1)]
            val: bool | float
            if raw.lower() == "true":
                val = True
            elif raw.lower() == "false":
                val = False
            else:
                val = float(raw)
            arr = table[col]
            if op == ">=":
                mask = pc.greater_equal(arr, val)
            elif op == "<=":
                mask = pc.less_equal(arr, val)
            elif op == ">":
                mask = pc.greater(arr, val)
            elif op == "<":
                mask = pc.less(arr, val)
            else:
                mask = pc.equal(arr, val)
            return table.filter(mask)
    return table


class PassiveScanner:
    """Batch multi-source anomaly screening across geometric patterns.

    Reads geometry ONCE per pattern (not per entity). Orders of magnitude
    faster than per-entity cross_pattern_profile for population screening.
    """

    def __init__(
        self,
        reader: GDSReader,
        sphere: Sphere,
        manifest: Manifest,
    ) -> None:
        self._reader = reader
        self._sphere = sphere
        self._manifest = manifest
        self._sources: list[ScanSource] = []

    def add_source(
        self,
        name: str,
        pattern_id: str,
        key_type: Literal["direct", "sibling", "composite", "chain"] | None = None,
        weight: float = 1.0,
        filter_expr: str | None = None,
    ) -> PassiveScanner:
        """Register an anomaly source.

        key_type auto-detected when None:
        - "direct" if pattern's entity_line matches home_line_id
        - "composite" if geometry keys contain separator
        - "chain" if points table has chain_keys column
        """
        if key_type is None:
            key_type = self._detect_key_type(pattern_id)
        self._sources.append(ScanSource(
            name=name,
            pattern_id=pattern_id,
            key_type=key_type,
            weight=weight,
            filter_expr=filter_expr,
        ))
        return self

    def add_borderline_source(
        self,
        name: str,
        pattern_id: str,
        rank_threshold: float = 80,
        weight: float = 1.0,
    ) -> PassiveScanner:
        """Register a borderline source — near-threshold, non-anomalous entities."""
        self._sources.append(ScanSource(
            name=name,
            pattern_id=pattern_id,
            key_type="borderline",
            weight=weight,
            filter_expr=f"delta_rank_pct >= {rank_threshold} AND is_anomaly = false",
            rank_threshold=rank_threshold,
        ))
        return self

    def add_points_source(
        self,
        name: str,
        line_id: str,
        rules: dict[str, tuple[str, float]],
        combine: Literal["AND", "OR"] = "AND",
        weight: float = 1.0,
    ) -> PassiveScanner:
        """Register a points-rule source — filter entities by column thresholds.

        rules: {column_name: (operator, value)} where operator is
               ">=", ">", "<=", "<", "==", "!=".
        combine: "AND" (all rules must match) or "OR" (any rule matches).
        """
        self._sources.append(ScanSource(
            name=name,
            pattern_id="",  # not used for points sources
            key_type="points",
            weight=weight,
            line_id=line_id,
            rules=rules,
            rules_combine=combine,
        ))
        return self

    def add_compound_source(
        self,
        name: str,
        geometry_pattern_id: str,
        line_id: str,
        rules: dict[str, tuple[str, float]],
        combine: Literal["AND", "OR"] = "AND",
        geometry_key_type: Literal["direct", "composite", "chain"] | None = None,
        geometry_filter_expr: str | None = None,
        chain_filter_expr: str | None = None,
        weight: float = 1.0,
    ) -> PassiveScanner:
        """Register a compound source — geometry expansion intersected with points rules.

        Entities must appear in BOTH the geometry source AND match the points rules.
        Typical use: "account in anomalous chain AND return_ratio >= 0.40".
        """
        if geometry_key_type is None:
            geometry_key_type = self._detect_key_type(geometry_pattern_id)
        self._sources.append(ScanSource(
            name=name,
            pattern_id=geometry_pattern_id,
            key_type="compound",
            weight=weight,
            filter_expr=geometry_filter_expr,
            line_id=line_id,
            rules=rules,
            rules_combine=combine,
            geometry_pattern_id=geometry_pattern_id,
            geometry_key_type=geometry_key_type,
            chain_filter_expr=chain_filter_expr,
        ))
        return self

    def auto_discover(
        self,
        home_line_id: str,
        include_borderline: bool = False,
        borderline_rank_threshold: float = 80,
    ) -> PassiveScanner:
        """Auto-discover all patterns related to home_line_id.

        Registers each discovered pattern as a source with auto-detected key_type.
        When include_borderline is True, also registers borderline sources for
        each direct pattern.
        """
        for pat_id in self._sphere.patterns:
            key_type = self._classify_for_line(pat_id, home_line_id)
            if key_type is not None:
                self._sources.append(ScanSource(
                    name=pat_id,
                    pattern_id=pat_id,
                    key_type=key_type,
                ))
                if include_borderline and key_type in ("direct", "sibling"):
                    self.add_borderline_source(
                        f"{pat_id}_borderline",
                        pat_id,
                        rank_threshold=borderline_rank_threshold,
                    )
        return self

    def scan(
        self,
        home_line_id: str,
        scoring: Literal["count", "weighted"] = "count",
        threshold: int = 1,
        top_n: int | None = None,
    ) -> ScanResult:
        """Execute batch scan across all registered sources.

        Returns entities above threshold, sorted by score descending.
        """
        t0 = time.time()

        # Phase 1: bulk geometry reads per source
        source_hits: dict[str, dict[str, ScanSourceHit]] = {}
        sources_summary: dict[str, int] = {}

        for source in self._sources:
            version = self._resolve_version(source.pattern_id) if source.pattern_id else None
            if version is None and source.key_type not in ("points",):
                continue

            if source.key_type in ("direct", "sibling"):
                hits = self._scan_direct(source, version)
            elif source.key_type == "composite":
                hits = self._scan_composite(source, version)
            elif source.key_type == "chain":
                hits = self._scan_chain(source, version)
            elif source.key_type == "borderline":
                hits = self._scan_direct(source, version)
            elif source.key_type == "points":
                hits = self._scan_points(source)
            elif source.key_type == "compound":
                hits = self._scan_compound(source, version)
            else:
                continue

            source_hits[source.name] = hits
            sources_summary[source.name] = len(hits)

        # Phase 2: multi-source aggregation
        all_entities: dict[str, dict[str, ScanSourceHit]] = defaultdict(dict)
        for src_name, hits in source_hits.items():
            for entity_key, hit in hits.items():
                all_entities[entity_key][src_name] = hit

        # Score and filter
        result_hits: list[ScanHit] = []
        for entity_key, per_source in all_entities.items():
            score = len(per_source)
            weighted = 0.0
            for _src_name, sh in per_source.items():
                if sh.anomaly_intensity > 0:
                    weighted += sh.anomaly_intensity
                else:
                    related = max(sh.related_count, 1)
                    weighted += sh.anomalous_count / related

            if scoring == "count" and score < threshold:
                continue
            if scoring == "weighted" and weighted < threshold:
                continue

            result_hits.append(ScanHit(
                primary_key=entity_key,
                score=score,
                weighted_score=round(weighted, 4),
                sources=per_source,
            ))

        # Sort by score desc, then weighted desc
        result_hits.sort(key=lambda h: (-h.score, -h.weighted_score))

        if top_n is not None:
            result_hits = result_hits[:top_n]

        # Total entities in home line
        total_entities = 0
        try:
            home_line = self._sphere.lines.get(home_line_id)
            if home_line:
                ver = home_line.current_version()
                pts = self._reader.read_points(home_line_id, ver)
                total_entities = pts.num_rows
        except Exception:
            pass

        elapsed = (time.time() - t0) * 1000
        return ScanResult(
            home_line_id=home_line_id,
            total_entities=total_entities,
            total_flagged=len(result_hits),
            hits=result_hits,
            sources_summary=sources_summary,
            elapsed_ms=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Private: per-source scan methods
    # ------------------------------------------------------------------

    def _scan_direct(
        self, source: ScanSource, version: int,
    ) -> dict[str, ScanSourceHit]:
        """Scan direct pattern — entity keys = geometry primary_keys."""
        filt = source.filter_expr or "is_anomaly = true"
        geo = self._reader.read_geometry(
            source.pattern_id, version,
            columns=["primary_key", "delta_norm"],
            filter=filt,
        )
        pattern = self._sphere.patterns[source.pattern_id]
        theta_norm = pattern.theta_norm
        result: dict[str, ScanSourceHit] = {}
        for pk, norm in zip(
            geo["primary_key"].to_pylist(),
            geo["delta_norm"].to_pylist(), strict=False,
        ):
            dn = float(norm) if norm is not None else 0.0
            intensity = dn / theta_norm if theta_norm > 0 else 1.0
            result[pk] = ScanSourceHit(
                anomalous_count=1,
                related_count=1,
                max_delta_norm=dn,
                anomaly_intensity=intensity,
            )
        return result

    def _scan_composite(
        self, source: ScanSource, version: int,
    ) -> dict[str, ScanSourceHit]:
        """Scan composite pattern — split composite keys to entity keys."""
        filt = source.filter_expr or "is_anomaly = true"
        geo = self._reader.read_geometry(
            source.pattern_id, version,
            columns=["primary_key", "delta_norm"],
            filter=filt,
        )
        # Detect separator from first PK
        sep = _COMPOSITE_SEP
        pks = geo["primary_key"].to_pylist()
        if pks:
            sample = pks[0]
            for candidate in (_COMPOSITE_SEP, "|", "→"):
                if candidate in sample:
                    sep = candidate
                    break

        entity_hits: dict[str, list[float]] = defaultdict(list)
        for pk, norm in zip(
            pks,
            geo["delta_norm"].to_pylist(), strict=False,
        ):
            norm_f = float(norm) if norm is not None else 0.0
            parts = pk.split(sep)
            for part in parts:
                if part:
                    entity_hits[part].append(norm_f)

        total = self._reader.count_geometry_rows(
            source.pattern_id, version,
        )

        result: dict[str, ScanSourceHit] = {}
        for ek, norms in entity_hits.items():
            result[ek] = ScanSourceHit(
                anomalous_count=len(norms),
                related_count=total,
                max_delta_norm=max(norms),
            )
        return result

    def _scan_chain(
        self, source: ScanSource, version: int,
    ) -> dict[str, ScanSourceHit]:
        """Scan chain pattern — reverse index chain_keys → entity keys."""
        entity_line_id = self._sphere.entity_line(source.pattern_id)
        if not entity_line_id:
            return {}

        line = self._sphere.lines.get(entity_line_id)
        if not line:
            return {}
        line_ver = line.current_version()

        pts = self._reader.read_points(entity_line_id, line_ver)
        if "chain_keys" not in pts.schema.names:
            return {}

        # Apply chain_filter_expr to chain points table if provided.
        # Supports: "col >= val", "col > val", "col <= val", "col < val", "col = val"
        if source.chain_filter_expr:
            pts = _apply_simple_filter(pts, source.chain_filter_expr)

        # Build chain_pk → entity keys mapping
        chain_to_entities: dict[str, list[str]] = defaultdict(list)
        for pk, ck in zip(
            pts["primary_key"].to_pylist(),
            pts["chain_keys"].to_pylist(), strict=False,
        ):
            if ck:
                for k in ck.split(","):
                    chain_to_entities[pk].append(k)

        # Read chains — None → anomalous only (default), "" → all chains
        filt: str | None = "is_anomaly = true" if source.filter_expr is None else (source.filter_expr or None)
        geo = self._reader.read_geometry(
            source.pattern_id, version,
            columns=["primary_key", "delta_norm"],
            filter=filt,
        )

        entity_hits: dict[str, list[float]] = defaultdict(list)
        for chain_pk, norm in zip(
            geo["primary_key"].to_pylist(),
            geo["delta_norm"].to_pylist(), strict=False,
        ):
            norm_f = float(norm) if norm is not None else 0.0
            for ek in chain_to_entities.get(chain_pk, []):
                entity_hits[ek].append(norm_f)

        total = self._reader.count_geometry_rows(
            source.pattern_id, version,
        )

        result: dict[str, ScanSourceHit] = {}
        for ek, norms in entity_hits.items():
            result[ek] = ScanSourceHit(
                anomalous_count=len(norms),
                related_count=total,
                max_delta_norm=max(norms),
            )
        return result

    _RULE_OPS = frozenset({">=", ">", "<=", "<", "==", "!="})

    def _scan_points(
        self, source: ScanSource,
    ) -> dict[str, ScanSourceHit]:
        """Scan points line with threshold rules."""
        import pyarrow.compute as pc

        if not source.line_id or not source.rules:
            return {}

        line = self._sphere.lines.get(source.line_id)
        if not line:
            return {}
        version = line.current_version()

        try:
            needed_cols = ["primary_key"] + list(source.rules.keys())
            pts = self._reader.read_points(source.line_id, version)
            available = set(pts.schema.names)
            proj_cols = [c for c in needed_cols if c in available]
            pts = pts.select(proj_cols)
        except Exception:
            return {}

        # Apply rules via pyarrow compute
        mask = None
        for col, (op, val) in source.rules.items():
            if op not in self._RULE_OPS:
                continue
            if col not in pts.schema.names:
                continue
            col_arr = pts[col]
            # Cast threshold to match column type
            import pyarrow as pa
            if pa.types.is_integer(col_arr.type):
                val = int(val)
            elif pa.types.is_floating(col_arr.type):
                val = float(val)
            if op == ">=":
                cond = pc.greater_equal(col_arr, val)
            elif op == ">":
                cond = pc.greater(col_arr, val)
            elif op == "<=":
                cond = pc.less_equal(col_arr, val)
            elif op == "<":
                cond = pc.less(col_arr, val)
            elif op == "==":
                cond = pc.equal(col_arr, val)
            elif op == "!=":
                cond = pc.not_equal(col_arr, val)
            else:
                continue
            cond = pc.fill_null(cond, False)
            if mask is None:
                mask = cond
            elif source.rules_combine == "AND":
                mask = pc.and_(mask, cond)
            else:
                mask = pc.or_(mask, cond)

        if mask is None:
            return {}

        filtered = pts.filter(mask)
        result: dict[str, ScanSourceHit] = {}
        for pk in filtered["primary_key"].to_pylist():
            result[pk] = ScanSourceHit(
                anomalous_count=1,
                related_count=0,
                max_delta_norm=0.0,
            )
        return result

    def _scan_compound(
        self, source: ScanSource, version: int,
    ) -> dict[str, ScanSourceHit]:
        """Scan compound source — geometry expansion intersected with points rules."""
        # Step 1: geometry expansion (reuse existing scan methods)
        geo_key_type = source.geometry_key_type or "direct"
        geo_source = ScanSource(
            name=source.name + "_geo",
            pattern_id=source.pattern_id,
            key_type=geo_key_type,
            filter_expr=source.filter_expr,
            chain_filter_expr=source.chain_filter_expr,
        )
        if geo_key_type in ("direct", "sibling"):
            geo_hits = self._scan_direct(geo_source, version)
        elif geo_key_type == "composite":
            geo_hits = self._scan_composite(geo_source, version)
        elif geo_key_type == "chain":
            geo_hits = self._scan_chain(geo_source, version)
        else:
            return {}

        if not geo_hits:
            return {}

        # Step 2: points filter
        points_source = ScanSource(
            name=source.name + "_pts",
            pattern_id="",
            key_type="points",
            line_id=source.line_id,
            rules=source.rules,
            rules_combine=source.rules_combine,
        )
        points_hits = self._scan_points(points_source)

        # Step 3: intersection
        result: dict[str, ScanSourceHit] = {}
        for pk in geo_hits:
            if pk in points_hits:
                result[pk] = geo_hits[pk]

        return result

    # ------------------------------------------------------------------
    # Private: key type detection
    # ------------------------------------------------------------------

    def _detect_key_type(
        self, pattern_id: str,
    ) -> Literal["direct", "composite", "chain"]:
        """Auto-detect key_type from sphere schema."""
        # Check all possible home lines
        entity_line_id = self._sphere.entity_line(pattern_id)
        if not entity_line_id:
            return "direct"

        # Check chain_keys
        line = self._sphere.lines.get(entity_line_id)
        if line and line.columns and any(c.name == "chain_keys" for c in line.columns):
            return "chain"

        # Check composite by sampling geometry
        version = self._resolve_version(pattern_id)
        if version is not None:
            try:
                sample = self._reader.read_geometry(
                    pattern_id, version,
                    columns=["primary_key"],
                    sample_size=1,
                )
                if sample.num_rows > 0:
                    pk = sample["primary_key"][0].as_py()
                    if _COMPOSITE_SEP in pk:
                        return "composite"
            except Exception:
                pass

        return "direct"

    def _classify_for_line(
        self,
        pattern_id: str,
        home_line_id: str,
    ) -> Literal["direct", "sibling", "composite", "chain"] | None:
        """Classify pattern relative to a specific home line."""
        entity_line_id = self._sphere.entity_line(pattern_id)

        # Direct: same anchor line
        if entity_line_id == home_line_id:
            return "direct"

        # Sibling: same source_id, different line_id (same key space)
        if entity_line_id in self._sphere.sibling_lines(home_line_id):
            return "sibling"

        if not entity_line_id:
            return None

        # Chain: has chain_keys column
        line = self._sphere.lines.get(entity_line_id)
        if line:
            if line.columns and any(c.name == "chain_keys" for c in line.columns):
                return "chain"

            # Composite: sample geometry for separator
            version = self._resolve_version(pattern_id)
            if version is not None:
                try:
                    sample = self._reader.read_geometry(
                        pattern_id, version,
                        columns=["primary_key"],
                        sample_size=1,
                    )
                    if sample.num_rows > 0:
                        pk = sample["primary_key"][0].as_py()
                        if _COMPOSITE_SEP in pk:
                            return "composite"
                except Exception:
                    pass

        return None

    def _resolve_version(self, pattern_id: str) -> int | None:
        """Resolve geometry version for a pattern."""
        return self._manifest.pattern_version(pattern_id)
