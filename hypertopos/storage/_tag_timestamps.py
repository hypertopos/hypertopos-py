# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tag timestamp abstraction — forward-compatible across Lance versions.

Lance 6.0 (current) does NOT carry a per-tag timestamp; the timestamp lives
on the underlying version. Tag → timestamp resolution costs an O(versions)
scan plus the O(1) `tags.get_version` lookup.

Lance 7.0 (forthcoming, beta as of 2026-05-11) adds native `tag_timestamp`
through #6364 which collapses the resolution to O(1) on a per-tag metadata
map. The public surface here stays the same — only the internal lookup
swaps when the bump lands.

Single public API; two implementations selected at import time on
`packaging.version.Version(_lance.__version__)`. Item 4 native MVCC
migration (sphere format 2.4 → 3.0) consumes this surface for calibration
epoch retention; no other module is allowed to read tag timestamps directly
to keep the swap site bounded.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import lance as _lance  # type: ignore[import-untyped]
from packaging.version import Version

_LANCE_VERSION = Version(_lance.__version__)
_NATIVE_TAG_TIMESTAMP_VERSION = Version("7.0.0")


def tag_timestamp(ds: _lance.LanceDataset, tag: str) -> datetime:
    """Return the datetime at which ``tag`` was created on ``ds``.

    Single public API. On Lance < 7.0 we look up the version that the tag
    points to and pull its commit timestamp from ``ds.versions()``. On
    Lance ≥ 7.0 we read the native ``tag_metadata.timestamp`` (when the
    Python binding ships it).

    Raises ``KeyError`` if the tag is unknown or the version is not in the
    dataset's version history (corrupt manifest case).
    """
    if _LANCE_VERSION >= _NATIVE_TAG_TIMESTAMP_VERSION:
        return _tag_timestamp_native(ds, tag)
    return _tag_timestamp_from_versions(ds, tag)


def cleanup_calibration_epochs(
    ds: _lance.LanceDataset,
    *,
    older_than: datetime,
    delete_unverified: bool = False,
) -> int:
    """Drop calibration-epoch tags older than ``older_than``, then run
    Lance's native ``cleanup_old_versions``. Returns the count of tags
    dropped.

    The native cleanup is invoked with ``error_if_tagged_old_versions=False``
    only AFTER the targeted tags are explicitly dropped — surviving tags
    pin their versions and are protected from cleanup. This keeps retention
    explicit at the call site rather than implicit in Lance's tag-honouring
    default.
    """
    dropped = 0
    for tag_name in list(ds.tags.list().keys()):
        try:
            ts = tag_timestamp(ds, tag_name)
        except KeyError:
            continue
        if ts < older_than:
            ds.tags.delete(tag_name)
            dropped += 1
    now = datetime.now(UTC) if older_than.tzinfo else datetime.now()
    age = now - older_than
    ds.cleanup_old_versions(
        older_than=age if age > timedelta(0) else None,
        delete_unverified=delete_unverified,
        error_if_tagged_old_versions=False,
    )
    return dropped


def _tag_timestamp_from_versions(
    ds: _lance.LanceDataset, tag: str,
) -> datetime:
    """Lance 6.0 path — derive tag timestamp from versions() scan.

    Reads ``tags.get_version(tag)`` then matches against ``ds.versions()``.
    O(N versions) per call; acceptable because callers (calibration epoch
    retention) sweep all tags once per build, not per query.
    """
    try:
        version = ds.tags.get_version(tag)
    except (ValueError, KeyError) as exc:
        raise KeyError(f"Tag {tag!r} not found on dataset: {exc}") from exc
    if version is None:
        raise KeyError(f"Tag {tag!r} not found on dataset")
    for v in ds.versions():
        if v.get("version") == version:
            ts = v.get("timestamp")
            if ts is None:
                raise KeyError(
                    f"Tag {tag!r} → version {version} has no timestamp",
                )
            return ts
    raise KeyError(
        f"Tag {tag!r} → version {version} not in dataset versions",
    )


def _tag_timestamp_native(ds: _lance.LanceDataset, tag: str) -> datetime:
    """Lance 7.0 path — read native per-tag timestamp.

    Stub: replaced when Lance 7.0 GA exposes the Python binding for
    ``tag_metadata.timestamp``. Until then, falls back to the versions
    scan path so the abstraction works on any 7.0-beta where the binding
    might lag the Rust crate.
    """
    tags_obj = ds.tags
    if hasattr(tags_obj, "get_metadata"):
        meta = tags_obj.get_metadata(tag)
        if meta is not None and "timestamp" in meta:
            return meta["timestamp"]
    return _tag_timestamp_from_versions(ds, tag)
