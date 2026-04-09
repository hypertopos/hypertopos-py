# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from hypertopos.builder.builder import GDSBuilder, IncrementalUpdateResult, RelationSpec
from hypertopos.builder.mapping import (
    LineMapping,
    MappingSpec,
    PatternMapping,
    RelationMapping,
    build_from_mapping,
    load_mapping,
)

__all__ = [
    "GDSBuilder",
    "IncrementalUpdateResult",
    "RelationSpec",
    "MappingSpec",
    "LineMapping",
    "PatternMapping",
    "RelationMapping",
    "build_from_mapping",
    "load_mapping",
]
