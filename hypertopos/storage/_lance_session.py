# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""One process-wide ``lance.Session`` for every read-path dataset open.

A bare ``lance.dataset()`` call starts with empty index and metadata caches,
so each re-open reloads scalar-index pages and manifests from disk. Opening
through the shared session keeps those caches warm across calls. Cache
entries are keyed by dataset URI and version, so one session can serve every
sphere in the process, and a re-open still sees the latest committed version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lance as _lance

_INDEX_CACHE_BYTES = 1 << 30
_METADATA_CACHE_BYTES = 256 << 20

# ``Session`` is public API, but the package stub does not re-export it.
_SESSION = _lance.Session(  # type: ignore[attr-defined]
    index_cache_size_bytes=_INDEX_CACHE_BYTES,
    metadata_cache_size_bytes=_METADATA_CACHE_BYTES,
)


def open_dataset(path: str | Path, **kwargs: Any) -> _lance.LanceDataset:
    """Open ``path`` with the shared session; ``kwargs`` go to ``lance.dataset``."""
    return _lance.dataset(str(path), session=_SESSION, **kwargs)
