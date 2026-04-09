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
            )
    elif args.command == "validate":
        from hypertopos.cli.build import run_validate
        run_validate(args.config)
    elif args.command == "info":
        from hypertopos.cli.info import run_info
        run_info(args.path)
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
