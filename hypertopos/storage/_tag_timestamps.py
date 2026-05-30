# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Tag timestamp abstraction — forward-compatible across Lance versions.

Pre-7.0 Lance does NOT carry a per-tag timestamp; the timestamp lives on the
underlying version. Tag → timestamp resolution costs an O(versions) scan plus
the O(1) `tags.get_version` lookup.

Lance 7.0 exposes the per-tag creation time directly on ``tags.list()``: each
entry is a dict carrying ``created_at`` (a tz-aware UTC ``datetime``). The
resolution collapses to an O(1) dict lookup. The public surface here stays the
same — only the internal lookup swaps based on whether the entry carries
``created_at``.

Single public API; the native path is preferred when ``tags.list()`` returns
``created_at`` for the tag and falls back to the version scan otherwise (tags
written by a pre-7.0 writer may not carry the field). Calibration-epoch
retention consumes this surface; no other module is allowed to read tag
timestamps directly to keep the swap site bounded.

Semantic note: the native path returns the **tag-creation** time, the version
scan returns the **version-commit** time. In practice these coincide — the
calibration epoch tag is created immediately after the calibration version is
committed — and tag-creation time is the correct semantic for "when the tag was
created" (matching the public docstring below).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import lance as _lance  # type: ignore[import-untyped]
from packaging.version import Version

_LANCE_VERSION = Version(_lance.__version__)
_NATIVE_TAG_TIMESTAMP_VERSION = Version("7.0.0")


def tag_timestamp(ds: _lance.LanceDataset, tag: str) -> datetime:
    """Return the datetime at which ``tag`` was created on ``ds``.

    Single public API. On Lance ≥ 7.0 we read the native ``created_at`` field
    carried on each ``tags.list()`` entry — an O(1) dict lookup. On older Lance
    (or for a tag written by a pre-7.0 writer that lacks ``created_at``) we look
    up the version that the tag points to and pull its commit timestamp from
    ``ds.versions()``.

    The returned datetime is tz-aware UTC on the native path and tz-naive on the
    version-scan path (Lance's ``versions()`` timestamps are tz-naive). Callers
    that compare across both paths must normalize (see
    :func:`cleanup_calibration_epochs`).

    Raises ``KeyError`` if the tag is unknown or the version is not in the
    dataset's version history (corrupt manifest case).
    """
    if _LANCE_VERSION >= _NATIVE_TAG_TIMESTAMP_VERSION:
        native = _tag_timestamp_native(ds, tag)
        if native is not None:
            return native
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
        # The native path returns tz-aware UTC; the version-scan path and a
        # caller-supplied cutoff may be tz-naive. Normalize both to the same
        # awareness before comparing — mixing them raises TypeError.
        if _as_comparable(ts, older_than) < _as_comparable(older_than, ts):
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


def _as_comparable(value: datetime, reference: datetime) -> datetime:
    """Return ``value`` with tz-awareness aligned to ``reference``.

    Python forbids comparing a tz-aware and a tz-naive ``datetime`` (raises
    ``TypeError``). When the two sides disagree, treat the naive one as UTC so
    both become aware. When both agree (both aware or both naive), ``value`` is
    returned unchanged.
    """
    if (value.tzinfo is None) == (reference.tzinfo is None):
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value  # value is aware, reference is naive — reference gets fixed


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


def _tag_timestamp_native(
    ds: _lance.LanceDataset, tag: str,
) -> datetime | None:
    """Lance 7.0 path — read the native per-tag ``created_at``.

    Each ``tags.list()`` entry is a dict carrying ``created_at`` (tz-aware UTC
    ``datetime``). Returns that value, or ``None`` to signal the caller to fall
    back to the version scan — either because the tag is absent from the listing
    or because the entry predates the field (tag written by a pre-7.0 writer).

    Unknown-tag handling is delegated to the version-scan fallback so the
    ``KeyError`` contract is raised from a single place.
    """
    entry = ds.tags.list().get(tag)
    if entry is None:
        return None
    created_at = entry.get("created_at")
    return created_at if isinstance(created_at, datetime) else None
