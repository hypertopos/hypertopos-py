from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa


EdgeTuple = tuple[str, float, float, str]


class _LazyEdgeMap:
    """Dict-like view over one direction of an AdjacencyIndex.

    Supports ``key in map``, ``map[key]``, ``map.get(key, default)``,
    ``map.keys()``, ``map.items()``, ``map.values()``, ``len(map)``,
    ``bool(map)``, and iteration — all without materializing the full
    edge list upfront.
    """
    __slots__ = ("_adj", "_kind")

    def __init__(self, adj: AdjacencyIndex, kind: str) -> None:
        self._adj = adj
        self._kind = kind

    def _idx_map(self) -> dict[str, int]:
        return self._adj._out_key_to_idx if self._kind == "out" else self._adj._in_key_to_idx

    def _mat(self, key: str) -> list[EdgeTuple]:
        if self._kind == "out":
            return self._adj._materialize_out(key)
        return self._adj._materialize_in(key)

    def __contains__(self, key: object) -> bool:
        return key in self._idx_map()

    def __bool__(self) -> bool:
        return bool(self._idx_map())

    def __len__(self) -> int:
        return len(self._idx_map())

    def __iter__(self) -> Iterator[str]:
        return iter(self._idx_map())

    def keys(self) -> Any:
        return self._idx_map().keys()

    def get(self, key: str, default: Any = None) -> Any:
        return self._mat(key) if key in self._idx_map() else default

    def __getitem__(self, key: str) -> list[EdgeTuple]:
        if key not in self._idx_map():
            raise KeyError(key)
        return self._mat(key)

    def items(self) -> Iterator[tuple[str, list[EdgeTuple]]]:
        for k in self._idx_map():
            yield k, self._mat(k)

    def values(self) -> Iterator[list[EdgeTuple]]:
        for k in self._idx_map():
            yield self._mat(k)


def temporal_bisect(
    edges: list[tuple[Any, ...]],
    ts_from: float | None,
    ts_to: float | None,
    ts_index: int = 1,
) -> list[tuple[Any, ...]]:
    """Bisect-based temporal filter on sorted edge lists."""
    from bisect import bisect_left, bisect_right
    if not edges:
        return []
    lo = bisect_left(edges, ts_from, key=lambda e: e[ts_index]) if ts_from is not None else 0
    hi = bisect_right(edges, ts_to, key=lambda e: e[ts_index]) if ts_to is not None else len(edges)
    return edges[lo:hi]


@dataclass
class AdjacencyIndex:
    """Directed adjacency index with O(1) per-key lookup.

    Internal storage is a pair of pyarrow Tables produced by
    ``Table.group_by(col).aggregate([(c, "list")])``. Per-key edge tuples
    are materialized lazily on first call to ``neighbors_out`` /
    ``neighbors_in`` and cached in-instance for the session.
    """
    _out_grouped: pa.Table = field(repr=False)
    _out_key_to_idx: dict[str, int] = field(repr=False)
    _in_grouped: pa.Table = field(repr=False)
    _in_key_to_idx: dict[str, int] = field(repr=False)
    _nodes: set[str] = field(repr=False)
    _edge_count: int = 0
    _pair_counts: dict[tuple[str, str], int] | None = field(default=None, repr=False)
    _out_cache: dict[str, list[EdgeTuple]] = field(default_factory=dict, repr=False)
    _in_cache: dict[str, list[EdgeTuple]] = field(default_factory=dict, repr=False)
    _out_distinct_count: dict[str, int] = field(default_factory=dict, repr=False)
    _in_distinct_count: dict[str, int] = field(default_factory=dict, repr=False)
    _out_max_amount_excl_self: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def _out(self) -> _LazyEdgeMap:
        return _LazyEdgeMap(self, "out")

    @property
    def _in(self) -> _LazyEdgeMap:
        return _LazyEdgeMap(self, "in")

    @classmethod
    def from_edge_lists(
        cls,
        from_keys: list[str],
        to_keys: list[str],
        timestamps: list[float],
        amounts: list[float],
        event_keys: list[str],
    ) -> AdjacencyIndex:
        if not from_keys:
            return cls._empty()
        import pyarrow as pa
        tbl = pa.table({
            "from_key": from_keys,
            "to_key": to_keys,
            "timestamp": timestamps,
            "amount": amounts,
            "event_key": event_keys,
        })
        return cls._from_table(tbl)

    @classmethod
    def from_lance(cls, reader: Any, pattern_id: str) -> AdjacencyIndex:
        table = reader.read_edges(pattern_id)
        if table.num_rows == 0:
            return cls._empty()
        return cls._from_table(table)

    @classmethod
    def _empty(cls) -> AdjacencyIndex:
        import pyarrow as pa
        empty_grouped = pa.table({
            "from_key": pa.array([], type=pa.string()),
            "to_key_list": pa.array([], type=pa.list_(pa.string())),
            "timestamp_list": pa.array([], type=pa.list_(pa.float64())),
            "amount_list": pa.array([], type=pa.list_(pa.float64())),
            "event_key_list": pa.array([], type=pa.list_(pa.string())),
        })
        empty_in = pa.table({
            "to_key": pa.array([], type=pa.string()),
            "from_key_list": pa.array([], type=pa.list_(pa.string())),
            "timestamp_list": pa.array([], type=pa.list_(pa.float64())),
            "amount_list": pa.array([], type=pa.list_(pa.float64())),
            "event_key_list": pa.array([], type=pa.list_(pa.string())),
        })
        return cls(
            _out_grouped=empty_grouped,
            _out_key_to_idx={},
            _in_grouped=empty_in,
            _in_key_to_idx={},
            _nodes=set(),
            _edge_count=0,
            _pair_counts={},  # eagerly empty (matches eager population in _from_table)
        )

    @classmethod
    def _from_table(cls, tbl: pa.Table) -> AdjacencyIndex:
        # Per-key list ordering = timestamp ascending. Achieved by sorting
        # the whole table by timestamp first, then relying on pyarrow's
        # `list` aggregator preserving input row order within each group
        # (verified pyarrow 23; if a future version changes this, callers
        # of neighbors_out / neighbors_in would silently see unsorted lists
        # and temporal_bisect would return wrong slices).
        sorted_tbl = tbl.sort_by("timestamp")

        # Two independent group_by aggregations on the same sorted_tbl —
        # parallelize via ThreadPoolExecutor. pyarrow's group_by releases the
        # GIL during the C++ aggregation kernel, so two concurrent threads
        # actually run in parallel and we cut serial 13s wall-clock to
        # max(per-direction) ~7s. Trade-off: peak RAM during the parallel
        # block is ~2x the per-direction allocation (both group_by results
        # live in memory simultaneously) — on AML HI-small (5M edges) ~1 GB.
        from concurrent.futures import ThreadPoolExecutor

        def _build_out_grouped() -> pa.Table:
            return sorted_tbl.group_by("from_key").aggregate([
                ("to_key", "list"),
                ("timestamp", "list"),
                ("amount", "list"),
                ("event_key", "list"),
            ])

        def _build_in_grouped() -> pa.Table:
            return sorted_tbl.group_by("to_key").aggregate([
                ("from_key", "list"),
                ("timestamp", "list"),
                ("amount", "list"),
                ("event_key", "list"),
            ])

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_out = executor.submit(_build_out_grouped)
            future_in = executor.submit(_build_in_grouped)
            out_grouped = future_out.result()
            in_grouped = future_in.result()

        out_keys = out_grouped["from_key"].to_pylist()
        out_key_to_idx = {k: i for i, k in enumerate(out_keys)}
        in_keys = in_grouped["to_key"].to_pylist()
        in_key_to_idx = {k: i for i, k in enumerate(in_keys)}

        import pyarrow.compute as pc

        # Single all-pairs aggregate — includes self-loops (matches legacy pair_counts semantic)
        all_pairs = sorted_tbl.group_by(["from_key", "to_key"]).aggregate(
            [("event_key", "count")]
        )
        fk_list = all_pairs["from_key"].to_pylist()
        tk_list = all_pairs["to_key"].to_pylist()
        count_list = all_pairs["event_key_count"].to_pylist()

        pair_counts_dict: dict[tuple[str, str], int] = {}
        out_distinct_count: dict[str, int] = {}
        in_distinct_count: dict[str, int] = {}
        for fk, tk, c in zip(fk_list, tk_list, count_list, strict=True):
            pair_counts_dict[(fk, tk)] = c
            if fk != tk:
                out_distinct_count[fk] = out_distinct_count.get(fk, 0) + 1
                in_distinct_count[tk] = in_distinct_count.get(tk, 0) + 1

        # Max amount per from_key, excluding self-loops, ignoring null amounts
        non_self_with_amount = sorted_tbl.filter(
            pc.and_(
                pc.not_equal(sorted_tbl["from_key"], sorted_tbl["to_key"]),
                pc.is_valid(sorted_tbl["amount"]),
            )
        )
        max_amount_grouped = non_self_with_amount.group_by("from_key").aggregate(
            [("amount", "max")]
        )
        out_max_amount_excl_self: dict[str, float] = dict(zip(
            max_amount_grouped["from_key"].to_pylist(),
            max_amount_grouped["amount_max"].to_pylist(),
            strict=True,
        ))

        nodes: set[str] = set(out_keys) | set(in_keys)
        return cls(
            _out_grouped=out_grouped,
            _out_key_to_idx=out_key_to_idx,
            _in_grouped=in_grouped,
            _in_key_to_idx=in_key_to_idx,
            _nodes=nodes,
            _edge_count=tbl.num_rows,
            _pair_counts=pair_counts_dict,
            _out_distinct_count=out_distinct_count,
            _in_distinct_count=in_distinct_count,
            _out_max_amount_excl_self=out_max_amount_excl_self,
        )

    def _materialize_out(self, key: str) -> list[EdgeTuple]:
        cached = self._out_cache.get(key)
        if cached is not None:
            return cached
        idx = self._out_key_to_idx.get(key)
        if idx is None:
            self._out_cache[key] = []
            return []
        to_keys = self._out_grouped["to_key_list"][idx].as_py()
        timestamps = self._out_grouped["timestamp_list"][idx].as_py()
        amounts = self._out_grouped["amount_list"][idx].as_py()
        event_keys = self._out_grouped["event_key_list"][idx].as_py()
        edges = list(zip(to_keys, timestamps, amounts, event_keys, strict=False))
        self._out_cache[key] = edges
        return edges

    def _materialize_in(self, key: str) -> list[EdgeTuple]:
        cached = self._in_cache.get(key)
        if cached is not None:
            return cached
        idx = self._in_key_to_idx.get(key)
        if idx is None:
            self._in_cache[key] = []
            return []
        from_keys = self._in_grouped["from_key_list"][idx].as_py()
        timestamps = self._in_grouped["timestamp_list"][idx].as_py()
        amounts = self._in_grouped["amount_list"][idx].as_py()
        event_keys = self._in_grouped["event_key_list"][idx].as_py()
        edges = list(zip(from_keys, timestamps, amounts, event_keys, strict=False))
        self._in_cache[key] = edges
        return edges

    def neighbors_out(
        self,
        key: str,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[EdgeTuple]:
        edges = self._materialize_out(key)
        if ts_from is None and ts_to is None:
            return edges
        return self._temporal_slice(edges, ts_from, ts_to)

    def neighbors_in(
        self,
        key: str,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[EdgeTuple]:
        edges = self._materialize_in(key)
        if ts_from is None and ts_to is None:
            return edges
        return self._temporal_slice(edges, ts_from, ts_to)

    def neighbors_out_window(
        self,
        key: str,
        ts_min: float | None = None,
        columns: tuple[str, ...] = ("to_key", "timestamp"),
    ) -> dict[str, list[Any]]:
        """Out-edges of `key`, optionally window-filtered by `ts >= ts_min`.

        Returns a dict mapping requested column name to a Python list. Caller
        iterates per-column without materializing per-row tuples. Unrequested
        columns (e.g. amount, event_key) are NEVER materialized.

        Faster than `neighbors_out` for callers that need only a subset of
        columns or a windowed slice — skips `as_py` on unused columns and uses
        pyarrow C++ filter for the window predicate. Always available cols:
        ``to_key``, ``timestamp``, ``amount``, ``event_key``.
        """
        valid = {"to_key", "timestamp", "amount", "event_key"}
        invalid = [c for c in columns if c not in valid]
        if invalid:
            raise ValueError(
                f"neighbors_out_window: unknown columns {invalid}; "
                f"valid: {sorted(valid)}",
            )
        idx = self._out_key_to_idx.get(key)
        if idx is None:
            return {col: [] for col in columns}
        import pyarrow.compute as pc
        ts_array = self._out_grouped["timestamp_list"][idx].values
        mask = pc.greater_equal(ts_array, ts_min) if ts_min is not None else None
        result: dict[str, list[Any]] = {}
        for col in columns:
            list_col_name = f"{col}_list"
            arr = self._out_grouped[list_col_name][idx].values
            if mask is not None:
                arr = arr.filter(mask)
            result[col] = arr.to_pylist()
        return result

    def neighbors_in_window(
        self,
        key: str,
        ts_min: float | None = None,
        columns: tuple[str, ...] = ("from_key", "timestamp"),
    ) -> dict[str, list[Any]]:
        """In-edges of `key`, optionally window-filtered by `ts >= ts_min`.

        Returns a dict mapping requested column name to a Python list. Caller
        iterates per-column without materializing per-row tuples. Unrequested
        columns (e.g. amount, event_key) are NEVER materialized.

        Faster than `neighbors_in` for callers that need only a subset of
        columns or a windowed slice — skips `as_py` on unused columns and uses
        pyarrow C++ filter for the window predicate. Always available cols:
        ``from_key``, ``timestamp``, ``amount``, ``event_key``.
        """
        valid = {"from_key", "timestamp", "amount", "event_key"}
        invalid = [c for c in columns if c not in valid]
        if invalid:
            raise ValueError(
                f"neighbors_in_window: unknown columns {invalid}; "
                f"valid: {sorted(valid)}",
            )
        idx = self._in_key_to_idx.get(key)
        if idx is None:
            return {col: [] for col in columns}
        import pyarrow.compute as pc
        ts_array = self._in_grouped["timestamp_list"][idx].values
        mask = pc.greater_equal(ts_array, ts_min) if ts_min is not None else None
        result: dict[str, list[Any]] = {}
        for col in columns:
            list_col_name = f"{col}_list"
            arr = self._in_grouped[list_col_name][idx].values
            if mask is not None:
                arr = arr.filter(mask)
            result[col] = arr.to_pylist()
        return result

    def neighbors_all(
        self,
        key: str,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[EdgeTuple]:
        return self.neighbors_out(key, ts_from, ts_to) + self.neighbors_in(key, ts_from, ts_to)

    def degree_out(self, key: str) -> int:
        idx = self._out_key_to_idx.get(key)
        if idx is None:
            return 0
        return len(self._out_grouped["to_key_list"][idx])

    def degree_in(self, key: str) -> int:
        idx = self._in_key_to_idx.get(key)
        if idx is None:
            return 0
        return len(self._in_grouped["from_key_list"][idx])

    def distinct_neighbors_out(self, key: str) -> int:
        """Count of distinct out-neighbors excluding self-loops. O(1)."""
        return self._out_distinct_count.get(key, 0)

    def distinct_neighbors_in(self, key: str) -> int:
        """Count of distinct in-neighbors excluding self-loops. O(1)."""
        return self._in_distinct_count.get(key, 0)

    def max_amount_out_excl_self(self, key: str) -> float:
        """Max amount across out-edges excluding self-loops, ignoring null amounts.
        Returns 0.0 if key has no qualifying edge. O(1)."""
        return self._out_max_amount_excl_self.get(key, 0.0)

    def all_nodes(self) -> set[str]:
        return self._nodes

    def all_edges(self) -> Iterator[tuple[str, str, float, float, str]]:
        for src in self._out_key_to_idx:
            for tgt, ts, amt, ek in self._materialize_out(src):
                yield src, tgt, ts, amt, ek

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return self._edge_count

    def pair_counts(self) -> dict[tuple[str, str], int]:
        if self._pair_counts is not None:
            return self._pair_counts
        counts: dict[tuple[str, str], int] = {}
        for src in self._out_key_to_idx:
            for tgt, _ts, _amt, _ek in self._materialize_out(src):
                key = (src, tgt)
                counts[key] = counts.get(key, 0) + 1
        self._pair_counts = counts
        return counts

    @staticmethod
    def _temporal_slice(
        edges: list[EdgeTuple], ts_from: float | None, ts_to: float | None,
    ) -> list[EdgeTuple]:
        return temporal_bisect(edges, ts_from, ts_to, ts_index=1)
