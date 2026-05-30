# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

import argparse
import sys


def main() -> None:
    """hypertopos CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hypertopos",
        description="hypertopos — Geometric Data Sphere toolkit",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # --- build ---
    build_p = sub.add_parser("build", help="Build a sphere from YAML config")
    build_config = build_p.add_mutually_exclusive_group(required=True)
    build_config.add_argument(
        "--config", metavar="FILE",
        help="Path to sphere.yaml (new format)",
    )
    build_config.add_argument(
        "--mapping", metavar="FILE",
        help="Path to gds_mapping.yaml (legacy format)",
    )
    build_p.add_argument(
        "--output", default=None, metavar="DIR",
        help="Output directory (overrides YAML setting)",
    )
    build_p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output directory",
    )
    build_p.add_argument(
        "--verbose", action="store_true",
        help="Print progress messages",
    )
    build_p.add_argument(
        "--no-temporal", action="store_true",
        help="Skip temporal snapshot build",
    )
    build_p.add_argument(
        "--no-chains", action="store_true",
        help="Skip chain extraction",
    )
    build_p.add_argument(
        "--no-edges", action="store_true",
        help="Skip edge table emission",
    )
    build_p.add_argument(
        "--label-aware-calibration", action="store_true",
        help=(
            "Enable label-aware per-dim calibration. No-op unless the "
            "sphere config declares a label_audit block selecting "
            "patterns to calibrate."
        ),
    )

    # --- validate ---
    validate_p = sub.add_parser(
        "validate", help="Validate sphere.yaml without building",
    )
    validate_p.add_argument(
        "--config", required=True, metavar="FILE",
        help="Path to sphere.yaml",
    )

    # --- info ---
    info_p = sub.add_parser("info", help="Print sphere summary")
    info_p.add_argument("path", help="Path to a built sphere directory")

    # --- sphere (cloud-ops verbs over a built sphere) ---
    sphere_p = sub.add_parser(
        "sphere", help="Cloud-ops verbs over a built sphere (health/validate/diff)",
    )
    sphere_sub = sphere_p.add_subparsers(dest="sphere_command", metavar="subcommand")

    health_p = sphere_sub.add_parser(
        "health", help="Health-check a built sphere (CI gate)",
    )
    health_p.add_argument("path", help="Path to a built sphere directory")
    health_p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit machine-readable JSON on stdout",
    )
    health_p.add_argument(
        "--exit-code-on-critical", action="store_true",
        help="Exit 2 when status is critical (HIGH-severity alerts present)",
    )

    svalidate_p = sphere_sub.add_parser(
        "validate", help="Structural integrity check over a built sphere",
    )
    svalidate_p.add_argument("path", help="Path to a built sphere directory")
    svalidate_p.add_argument(
        "--strict", action="store_true",
        help="Promote calibration / dimension-quality warnings to errors",
    )
    svalidate_p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit machine-readable JSON on stdout",
    )

    sdiff_p = sphere_sub.add_parser(
        "diff", help="Diff two built spheres (pattern inventory + calibration drift)",
    )
    sdiff_p.add_argument("old", help="Path to the old built sphere directory")
    sdiff_p.add_argument("new", help="Path to the new built sphere directory")
    sdiff_p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit machine-readable JSON on stdout",
    )

    singest_p = sphere_sub.add_parser(
        "ingest",
        help="Incrementally append a new-/changed-entities table to a pattern",
    )
    singest_p.add_argument("path", help="Path to a built sphere directory")
    singest_p.add_argument(
        "--points", required=True, metavar="FILE",
        help=(
            "New-/changed-entities table (.arrow/.arrows, .parquet/.pq, "
            ".csv/.csv.gz) with a primary_key column"
        ),
    )
    singest_p.add_argument(
        "--pattern", default=None, metavar="ID",
        help=(
            "Pattern to ingest into. Optional when the sphere has exactly one "
            "pattern; required otherwise"
        ),
    )
    singest_p.add_argument(
        "--recalibrate", default="auto", choices=("auto", "force", "never"),
        help="Recalibration policy passed to incremental_update (default: auto)",
    )
    singest_p.add_argument(
        "--reindex", action="store_true",
        help="Force an ANN index rebuild so appended rows are immediately indexed",
    )
    singest_p.add_argument(
        "--finalize", action="store_true",
        help=(
            "Recompute the global delta_rank_pct and rebuild the ANN index once "
            "after the append (use at the end of a batched ingest session)"
        ),
    )
    singest_p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit machine-readable JSON on stdout",
    )

    args = parser.parse_args()

    if args.command == "build":
        if args.mapping:
            # Legacy path: delegate to builder.mapping
            _cmd_build_legacy(args.mapping, args.output)
        else:
            from hypertopos.cli.build import run_build
            run_build(
                args.config, args.output, args.force, args.verbose,
                no_temporal=args.no_temporal, no_chains=args.no_chains,
                no_edges=args.no_edges,
                label_aware_calibration=args.label_aware_calibration,
            )
    elif args.command == "validate":
        from hypertopos.cli.build import run_validate
        run_validate(args.config)
    elif args.command == "info":
        from hypertopos.cli.info import run_info
        run_info(args.path)
    elif args.command == "sphere":
        if args.sphere_command == "health":
            from hypertopos.cli.sphere_ops import run_sphere_health
            run_sphere_health(
                args.path, args.as_json, args.exit_code_on_critical,
            )
        elif args.sphere_command == "validate":
            from hypertopos.cli.sphere_ops import run_sphere_validate
            run_sphere_validate(args.path, args.strict, args.as_json)
        elif args.sphere_command == "diff":
            from hypertopos.cli.sphere_ops import run_sphere_diff
            run_sphere_diff(args.old, args.new, args.as_json)
        elif args.sphere_command == "ingest":
            from hypertopos.cli.sphere_ops import run_sphere_ingest
            run_sphere_ingest(
                args.path, args.points, args.pattern, args.recalibrate,
                args.reindex, args.finalize, args.as_json,
            )
        else:
            sphere_p.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_build_legacy(mapping: str, output: str | None) -> None:
    """Legacy build path using gds_mapping.yaml format."""
    from pathlib import Path

    from hypertopos.builder.mapping import build_from_mapping, load_mapping

    mapping_path = Path(mapping).resolve()
    try:
        spec = load_mapping(mapping_path)
        base_dir = mapping_path.parent
        out = build_from_mapping(spec, base_dir=base_dir, output_path=output)
        print(f"Built: {out}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
