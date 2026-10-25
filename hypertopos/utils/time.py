# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Bulk conversion of timestamp-like columns to epoch seconds.

Naive timestamps (no zone offset) are interpreted as UTC everywhere.
"""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

# Normalise to zero-padded ``YYYY-MM-DD`` so the whole-array ISO cast succeeds.
_DATE_REWRITES = (
    (r"^(\d{4})/(\d{1,2})/(\d{1,2})", r"\1-\2-\3"),
    (r"^(\d{4})-(\d)-", r"\1-0\2-"),
    (r"^(\d{4})-(\d{2})-(\d)(\D|$)", r"\1-\2-0\3\4"),
    # Sub-microsecond fractions: the result has microsecond resolution anyway.
    (r"(\.\d{6})\d+", r"\1"),
)
_ZONE_SUFFIX = r"(Z|[+-]\d{2}:?\d{2})$"
# Fallback formats for strings the ISO cast rejects.
_FALLBACK_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
)
_MICROS = pa.timestamp("us")
_MICROS_UTC = pa.timestamp("us", tz="UTC")


def _micros(ts: pa.Array) -> pa.Array:
    return pc.cast(pc.cast(ts, _MICROS), pa.int64())


def _parse_strings(strings: pa.Array, target: pa.DataType) -> pa.Array:
    """Parse ISO-like strings to int64 microseconds; unparseable values become null."""
    try:
        return _micros(pc.cast(strings, target))
    except pa.ArrowInvalid:
        pass
    parsed = pc.coalesce(
        *(
            pc.strptime(strings, format=fmt, unit="us", error_is_null=True)
            for fmt in _FALLBACK_FORMATS
        )
    )
    # strptime rolls a nonexistent day over (2022-02-30 -> 03-01). Only days
    # 29-31 can roll, so only those rows are read back and compared.
    suspect = pc.fill_null(
        pc.and_(
            pc.is_valid(parsed),
            pc.greater_equal(pc.utf8_slice_codeunits(strings, 8, 10), "29"),
        ),
        False,
    )
    if pc.any(suspect).as_py():
        rolled = pc.not_equal(
            pc.strftime(parsed.filter(suspect), format="%Y-%m-%d"),
            pc.utf8_slice_codeunits(strings.filter(suspect), 0, 10),
        )
        bad = pc.replace_with_mask(suspect, suspect, rolled)
        parsed = pc.if_else(bad, pa.scalar(None, parsed.type), parsed)
    micros = _micros(parsed)
    pending = pc.and_(pc.is_valid(strings), pc.is_null(micros))
    if not pc.any(pending).as_py():
        return micros
    # Only the values no vectorised path could parse are cast one by one.
    patch = []
    for value in strings.filter(pending).to_pylist():
        try:
            patch.append(_micros(pc.cast(pa.array([value]), target))[0].as_py())
        except pa.ArrowInvalid:
            patch.append(None)
    return pc.replace_with_mask(micros, pending, pa.array(patch, pa.int64()))


def to_epoch_seconds(
    values: pa.Array | pa.ChunkedArray,
    *,
    errors: Literal["raise", "null"] = "raise",
) -> pa.DoubleArray:
    """Convert timestamps, dates, epoch numbers or date strings to float64 epoch seconds.

    Accepts timestamp and date arrays (any unit, naive or zoned), integer/float
    arrays (already seconds) and string arrays (ISO 8601 with ``-`` or ``/``
    date separators, optional time, optional ``Z`` / ``±HH:MM`` offset), plain
    or dictionary-encoded. Nulls and blank strings become null. A non-blank
    string that cannot be parsed raises ``ValueError`` naming it, or becomes
    null with ``errors="null"``.
    """
    arr = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode()
    if pa.types.is_date(arr.type):
        arr = pc.cast(arr, pa.timestamp("ms"))
    if pa.types.is_timestamp(arr.type):
        micros = _micros(arr)
    elif pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type):
        return pc.cast(arr, pa.float64())
    elif pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        null_str = pa.scalar(None, arr.type)
        arr = pc.utf8_trim_whitespace(arr)
        arr = pc.if_else(pc.equal(arr, ""), null_str, arr)
        norm = arr
        for pattern, replacement in _DATE_REWRITES:
            norm = pc.replace_substring_regex(norm, pattern=pattern, replacement=replacement)
        zoned = pc.fill_null(pc.match_substring_regex(norm, pattern=_ZONE_SUFFIX), False)
        micros = pc.coalesce(
            _parse_strings(pc.if_else(zoned, norm, null_str), _MICROS_UTC),
            _parse_strings(pc.if_else(zoned, null_str, norm), _MICROS),
        )
        if errors == "raise":
            unparsed = pc.and_(pc.is_valid(arr), pc.is_null(micros))
            if pc.any(unparsed).as_py():
                bad = arr.filter(unparsed)[0].as_py()
                raise ValueError(f"cannot parse timestamp {bad!r}")
    else:
        raise TypeError(f"unsupported timestamp column type: {arr.type}")
    return pc.divide(pc.cast(micros, pa.float64()), 1e6)
