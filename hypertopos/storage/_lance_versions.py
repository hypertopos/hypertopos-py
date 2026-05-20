# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Map calibration epoch N to internal Lance dataset version for native MVCC geometry.

Sphere format 3.0 stores geometry as a single ``geometry/<pattern_id>/data.lance``
dataset where each calibration epoch lands as a new internal Lance version
tagged ``epoch_<N>``. This module concentrates the epoch <-> Lance-version
mapping; callers read or write epochs through this surface only.
"""
from __future__ import annotations

import lance as _lance


def epoch_tag(epoch: int) -> str:
    """Canonical tag name for calibration epoch ``epoch``."""
    return f"epoch_{epoch}"


def tag_epoch(ds: _lance.LanceDataset, epoch: int) -> None:
    """Tag the dataset's current ``latest_version`` as ``epoch_<epoch>``.

    Idempotent on the same (tag, version) pair via Lance's tag-update path:
    if the tag already exists, it is moved to the current latest_version.
    """
    name = epoch_tag(epoch)
    version = int(ds.latest_version)
    existing = ds.tags.list().get(name)
    if existing is not None:
        if int(existing.get("version", -1)) == version:
            return
        ds.tags.update(name, version)
        return
    ds.tags.create(name, version)
