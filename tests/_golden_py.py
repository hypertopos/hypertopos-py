# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Golden-output snapshot helpers for the hypertopos-py test suite.

A golden test turns a call-site result into a canonical JSON document and
compares it with the file committed under ``tests/golden/``. Set
``HYPERTOPOS_UPDATE_GOLDEN=1`` to rewrite the files instead of comparing.
See ``tests/golden/README.md`` for the rules.

The same helpers live in hypertopos-mcp as ``tests/_golden_mcp.py``; the two
copies stay separate because each package ships as its own repository.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

UPDATE_ENV = "HYPERTOPOS_UPDATE_GOLDEN"
GOLDEN_DIR = Path(__file__).parent / "golden"

# Significant digits kept when a float is written to a golden file.
SIG_DIGITS = 10
# Relative / absolute tolerance used when comparing floats.
REL_TOL = 1e-6
ABS_TOL = 1e-9
# Golden floats printed with at most this many decimals are treated as
# already rounded by the code under test; a difference of one unit in the
# last printed decimal is then accepted (a rounding-boundary flip). The
# printed form drops trailing zeros (0.0500 prints as 0.05), so the quantum
# is never taken coarser than 4 decimals, the coarsest common output rounding.
_PRE_ROUNDED_MAX_DECIMALS = 6
_PRE_ROUNDED_MIN_DECIMALS = 4

# Keys whose values change on every run (wall-clock timings, build/refresh
# "now" timestamps). They are removed at any depth before comparison.
VOLATILE_KEYS = frozenset(
    {
        "elapsed_ms",
        "elapsed_s",
        "elapsed_seconds",
        "computed_at",
        "created_at",
        "updated_at",
        "last_refresh_at",
        "built_at",
        "generated_at",
    }
)

_MAX_DIFF_LINES = 20


def canonicalize(
    obj: Any,
    *,
    drop_keys: Iterable[str] = (),
    path_tokens: dict[str, str] | None = None,
) -> Any:
    """Return a JSON-ready, platform-stable copy of ``obj``.

    - dataclasses become dicts, tuples become lists, sets become sorted lists;
    - numpy scalars / arrays become Python numbers / lists;
    - floats are rounded to ``SIG_DIGITS`` significant digits, ``-0.0``
      becomes ``0.0``, NaN / infinities become the strings ``"NaN"`` /
      ``"Infinity"`` / ``"-Infinity"``;
    - datetimes become ISO strings;
    - keys in ``VOLATILE_KEYS`` or ``drop_keys`` are removed at any depth;
    - every occurrence of a ``path_tokens`` key inside a string is replaced
      by its token (e.g. an absolute sphere path by ``<SPHERE>``).

    List order is kept: ranking order is part of the contract.
    """
    drop = VOLATILE_KEYS | frozenset(drop_keys)
    tokens = path_tokens or {}
    return _canon(obj, drop, tokens)


def _canon(obj: Any, drop: frozenset[str], tokens: dict[str, str]) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _canon(v, drop, tokens) for k, v in obj.items() if str(k) not in drop}
    if isinstance(obj, (list, tuple)):
        return [_canon(v, drop, tokens) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [_canon(v, drop, tokens) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, np.ndarray):
        return _canon(obj.tolist(), drop, tokens)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _canon_float(float(obj))
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        obj = str(obj)
    if isinstance(obj, str):
        for raw, token in tokens.items():
            for form in _path_forms(raw):
                obj = obj.replace(form, token)
        return obj
    if obj is None:
        return None
    raise TypeError(f"golden: cannot canonicalize {type(obj).__name__}: {obj!r}")


def _path_forms(raw: str) -> list[str]:
    forms = {raw, raw.replace("\\", "/"), raw.replace("/", "\\")}
    forms |= {f.replace("\\", "\\\\") for f in list(forms)}
    return sorted(forms, key=len, reverse=True)


def _canon_float(x: float) -> float | str:
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0.0:
        return 0.0
    return float(f"{x:.{SIG_DIGITS}g}")


def _float_tolerance(expected: float, actual: float) -> float:
    tol = max(REL_TOL * max(abs(expected), abs(actual)), ABS_TOL)
    text = repr(expected)
    match = re.fullmatch(r"-?\d+\.(\d+)", text)
    if match and len(match.group(1)) <= _PRE_ROUNDED_MAX_DECIMALS:
        decimals = max(len(match.group(1)), _PRE_ROUNDED_MIN_DECIMALS)
        tol = max(tol, 1.01 * 10.0**-decimals)
    return tol


def diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Return human-readable differences between two canonical documents."""
    out: list[str] = []
    _diff(expected, actual, path, out)
    return out


def _diff(expected: Any, actual: Any, path: str, out: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}"
            if key not in actual:
                out.append(f"{sub}: missing in actual (expected {_short(expected[key])})")
            elif key not in expected:
                out.append(f"{sub}: unexpected in actual ({_short(actual[key])})")
            else:
                _diff(expected[key], actual[key], sub, out)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(expected)} -> {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual, strict=False)):
            _diff(e, a, f"{path}[{i}]", out)
        return
    if (
        isinstance(expected, float)
        and isinstance(actual, float)
        and math.isclose(expected, actual, rel_tol=0.0, abs_tol=_float_tolerance(expected, actual))
    ):
        return
    if type(expected) is not type(actual) or expected != actual:
        out.append(f"{path}: expected {_short(expected)}, got {_short(actual)}")


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 120 else text[:117] + "..."


def dumps(doc: Any) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _update_hint() -> str:
    current = os.environ.get("PYTEST_CURRENT_TEST", "")
    target = current.split(" ")[0] if current else "<golden test file>"
    return f"regenerate with: {UPDATE_ENV}=1 python -m pytest {target}"


def check_golden(name: str, payload: Any, *, golden_dir: Path = GOLDEN_DIR) -> None:
    """Compare ``payload`` (already canonical) with ``golden_dir/<name>.json``.

    With ``HYPERTOPOS_UPDATE_GOLDEN=1`` the file is (re)written instead.
    """
    path = golden_dir / f"{name}.json"
    # Round-trip through JSON so the comparison sees exactly what a file holds.
    actual = json.loads(dumps(payload))
    if os.environ.get(UPDATE_ENV) == "1":
        golden_dir.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(dumps(actual))
        return
    if not path.exists():
        raise AssertionError(f"golden file {path.name} is missing; {_update_hint()}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    problems = diff(expected, actual)
    if problems:
        shown = problems[:_MAX_DIFF_LINES]
        more = len(problems) - len(shown)
        lines = [f"golden mismatch in {path.name} ({len(problems)} difference(s)):"]
        lines += [f"  {p}" for p in shown]
        if more:
            lines.append(f"  ... and {more} more")
        lines.append(
            f"If the change is intended, {_update_hint()} and explain the golden diff "
            "in the PR; otherwise it is a regression."
        )
        raise AssertionError("\n".join(lines))
