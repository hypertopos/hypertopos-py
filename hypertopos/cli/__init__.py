# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""hypertopos CLI package.

Re-exports ``main`` for backward compatibility:
    from hypertopos.cli import main
"""
from __future__ import annotations

from hypertopos.cli.main import main

__all__ = ["main"]
