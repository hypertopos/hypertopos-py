"""Library code never draws from a global RNG: results must not depend on hidden state."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np

_PACKAGE = Path(__file__).resolve().parents[1] / "hypertopos"
_STDLIB_DRAWS = {"sample", "randint", "choice", "choices", "shuffle", "random", "uniform", "gauss"}
_NUMPY_LEGACY = {
    "rand",
    "randn",
    "randint",
    "choice",
    "shuffle",
    "permutation",
    "random",
    "uniform",
    "normal",
    "seed",
}


def _global_draws() -> list[str]:
    found = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "random"
                and node.func.attr in _STDLIB_DRAWS
            ):
                found.append(f"{path.relative_to(_PACKAGE)}:{node.lineno} random.{node.func.attr}")
            if (
                isinstance(owner, ast.Attribute)
                and owner.attr == "random"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "np"
                and node.func.attr in _NUMPY_LEGACY
            ):
                found.append(
                    f"{path.relative_to(_PACKAGE)}:{node.lineno} np.random.{node.func.attr}"
                )
    return found


def test_no_draws_from_global_random_state() -> None:
    assert _global_draws() == []


def test_centroid_member_samples_do_not_depend_on_global_random_state() -> None:
    from hypertopos.engine.geometry import GDSEngine

    engine = GDSEngine(storage=None, cache=None)
    deltas = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = ["A"] * 12 + ["B"] * 8
    keys = [f"K{i}" for i in range(20)]

    random.seed(1)
    first = engine.compute_centroid_map(deltas, labels, entity_keys=keys)
    random.seed(2)
    second = engine.compute_centroid_map(deltas, labels, entity_keys=keys)

    def samples(result):
        return {g["key"]: g["member_samples"] for g in result["group_centroids"]}

    assert samples(first) == samples(second)
