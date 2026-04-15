from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

EdgeTuple = tuple[str, float, float, str]


def temporal_bisect(
    edges: list[tuple],
    ts_from: float | None,
    ts_to: float | None,
    ts_index: int = 1,
) -> list[tuple]:
    """Bisect-based temporal filter on sorted edge lists."""
    from bisect import bisect_left, bisect_right
    if not edges:
        return []
    lo = bisect_left(edges, ts_from, key=lambda e: e[ts_index]) if ts_from is not None else 0
    hi = bisect_right(edges, ts_to, key=lambda e: e[ts_index]) if ts_to is not None else len(edges)
    return edges[lo:hi]


@dataclass
class AdjacencyIndex:
    _out: dict[str, list[EdgeTuple]] = field(repr=False)
    _in: dict[str, list[EdgeTuple]] = field(repr=False)
    _nodes: set[str] = field(repr=False)
    _edge_count: int = 0

    @classmethod
    def from_edge_lists(
        cls,
        from_keys: list[str],
        to_keys: list[str],
        timestamps: list[float],
        amounts: list[float],
        event_keys: list[str],
    ) -> AdjacencyIndex:
        out: dict[str, list[EdgeTuple]] = defaultdict(list)
        inv: dict[str, list[EdgeTuple]] = defaultdict(list)
        nodes: set[str] = set()
        n = len(from_keys)
        for i in range(n):
            fk, tk = from_keys[i], to_keys[i]
            ts, amt, ek = timestamps[i], amounts[i], event_keys[i]
            out[fk].append((tk, ts, amt, ek))
            inv[tk].append((fk, ts, amt, ek))
            nodes.add(fk)
            nodes.add(tk)
        for v in out.values():
            v.sort(key=lambda e: e[1])
        for v in inv.values():
            v.sort(key=lambda e: e[1])
        return cls(_out=dict(out), _in=dict(inv), _nodes=nodes, _edge_count=n)

    @classmethod
    def from_lance(cls, reader: Any, pattern_id: str) -> AdjacencyIndex:
        table = reader.read_edges(pattern_id)
        if table.num_rows == 0:
            return cls(_out={}, _in={}, _nodes=set(), _edge_count=0)
        from_keys = table["from_key"].to_pylist()
        to_keys = table["to_key"].to_pylist()
        timestamps = table["timestamp"].to_pylist()
        amounts = table["amount"].to_pylist()
        event_keys = table["event_key"].to_pylist()
        return cls.from_edge_lists(from_keys, to_keys, timestamps, amounts, event_keys)

    def neighbors_out(self, key: str, ts_from: float | None = None, ts_to: float | None = None) -> list[EdgeTuple]:
        edges = self._out.get(key, [])
        if ts_from is None and ts_to is None:
            return edges
        return self._temporal_slice(edges, ts_from, ts_to)

    def neighbors_in(self, key: str, ts_from: float | None = None, ts_to: float | None = None) -> list[EdgeTuple]:
        edges = self._in.get(key, [])
        if ts_from is None and ts_to is None:
            return edges
        return self._temporal_slice(edges, ts_from, ts_to)

    def neighbors_all(self, key: str, ts_from: float | None = None, ts_to: float | None = None) -> list[EdgeTuple]:
        return self.neighbors_out(key, ts_from, ts_to) + self.neighbors_in(key, ts_from, ts_to)

    def degree_out(self, key: str) -> int:
        return len(self._out.get(key, []))

    def degree_in(self, key: str) -> int:
        return len(self._in.get(key, []))

    def all_nodes(self) -> set[str]:
        return self._nodes

    def all_edges(self) -> Iterator[tuple[str, str, float, float, str]]:
        for src, edges in self._out.items():
            for tgt, ts, amt, ek in edges:
                yield src, tgt, ts, amt, ek

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return self._edge_count

    @staticmethod
    def _temporal_slice(edges: list[EdgeTuple], ts_from: float | None, ts_to: float | None) -> list[EdgeTuple]:
        return temporal_bisect(edges, ts_from, ts_to, ts_index=1)
