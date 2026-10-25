# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Run allocation-heavy build steps with the cyclic garbage collector paused."""

from __future__ import annotations

import functools
import gc
from collections.abc import Callable
from typing import Any, cast


def without_cyclic_gc[F: Callable[..., Any]](fn: F) -> F:
    """Run ``fn`` with the cyclic garbage collector paused.

    Steps that allocate millions of acyclic tuples (adjacency lists, DFS
    frames) pay for every generation-2 pass rescanning all of them.
    Reference counting still frees memory; the previous collector state is
    restored afterwards.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            return fn(*args, **kwargs)
        finally:
            if was_enabled:
                gc.enable()

    return cast("F", wrapper)
