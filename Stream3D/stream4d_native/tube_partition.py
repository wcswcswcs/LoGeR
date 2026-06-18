from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .signed_tube_graph import TubeGraphEdge


@dataclass
class TubePartitionResult:
    components: list[list[int]]
    diagnostics: dict[str, Any]


def partition_tube_graph(tube_ids: list[int], edges: list[TubeGraphEdge]) -> TubePartitionResult:
    parent = {int(tube_id): int(tube_id) for tube_id in tube_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        if int(edge.sign) > 0:
            union(int(edge.tube_i), int(edge.tube_j))
    groups: dict[int, list[int]] = {}
    for tube_id in parent:
        groups.setdefault(find(tube_id), []).append(int(tube_id))
    components = [sorted(values) for values in groups.values()]
    components.sort(key=lambda values: (len(values), values[0]), reverse=True)
    return TubePartitionResult(
        components=components,
        diagnostics={
            "tube_count": int(len(tube_ids)),
            "positive_edge_count": int(sum(1 for e in edges if int(e.sign) > 0)),
            "component_count": int(len(components)),
            "largest_component_size": int(max((len(c) for c in components), default=0)),
        },
    )


def filter_edges_by_pair_evidence(
    edges: list[TubeGraphEdge],
    pair_evidence: dict[tuple[int, int], dict[str, int]],
    *,
    mode: str = "negative_majority",
) -> list[TubeGraphEdge]:
    """Filter positive edges using image-space visible-outside negative evidence."""

    if mode not in {"negative_majority", "negative_strict"}:
        raise ValueError(f"unsupported negative evidence filter mode: {mode}")
    kept: list[TubeGraphEdge] = []
    for edge in edges:
        pair = tuple(sorted((int(edge.tube_i), int(edge.tube_j))))
        counts = pair_evidence.get(pair, {})
        same = int(counts.get("same_mask_count", 1))
        negative = int(counts.get("visible_outside_negative_count", 0))
        if mode == "negative_majority":
            keep = same > negative
        else:
            keep = negative == 0
        if keep:
            kept.append(edge)
    return kept


def filter_edges_by_mutual_topk(edges: list[TubeGraphEdge], *, top_k: int = 1) -> list[TubeGraphEdge]:
    """Keep edges that are among the top-k scored neighbors for both endpoints."""

    k = int(top_k)
    if k <= 0:
        raise ValueError("top_k must be positive")
    ranked: dict[int, list[TubeGraphEdge]] = {}
    for edge in edges:
        ranked.setdefault(int(edge.tube_i), []).append(edge)
        ranked.setdefault(int(edge.tube_j), []).append(edge)
    allowed: dict[int, set[tuple[int, int]]] = {}
    for tube_id, tube_edges in ranked.items():
        ordered = sorted(
            tube_edges,
            key=lambda edge: (-float(edge.score), float(edge.distance), int(edge.tube_i), int(edge.tube_j)),
        )
        allowed[int(tube_id)] = {
            tuple(sorted((int(edge.tube_i), int(edge.tube_j)))) for edge in ordered[:k]
        }
    kept: list[TubeGraphEdge] = []
    for edge in edges:
        pair = tuple(sorted((int(edge.tube_i), int(edge.tube_j))))
        if pair in allowed.get(int(edge.tube_i), set()) and pair in allowed.get(int(edge.tube_j), set()):
            kept.append(edge)
    return kept


def filter_edges_by_min_score(edges: list[TubeGraphEdge], *, min_score: float) -> list[TubeGraphEdge]:
    """Keep positive edges above a normalized score threshold."""

    threshold = float(min_score)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_score must be in [0, 1]")
    return [edge for edge in edges if float(edge.score) >= threshold]
