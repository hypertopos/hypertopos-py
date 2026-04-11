# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from hypertopos.builder import GDSBuilder, RelationSpec
from hypertopos.navigation.navigator import GDSError, GDSNavigationError, GDSNavigator, SimilarityResult
from hypertopos.sphere import HyperSession, HyperSphere

__all__ = [
    "HyperSphere",
    "HyperSession",
    "GDSNavigator",
    "GDSError",
    "GDSNavigationError",
    "SimilarityResult",
    "GDSBuilder",
    "RelationSpec",
]
__version__ = "0.2.2"
