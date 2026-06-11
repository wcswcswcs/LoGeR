from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .measurement_bank import MeasurementBank
from .signed_boundary_evidence import SignedBoundaryEvidence
from .signed_surfel_graph import SignedSurfelGraph


@dataclass
class SignedPartitionResult:
    scene: str
    mode: str
    components: list[np.ndarray]
    fringe_components: list[np.ndarray]
    diagnostics: dict[str, Any]


def _mask_votes(bank: MeasurementBank, surfels: np.ndarray, max_votes: int) -> list[tuple[int, int, float]]:
    votes: dict[tuple[int, int], float] = {}
    frame_ids = np.asarray(bank.frame_ids, dtype=np.int64)
    target_mask_id = np.asarray(bank.target_mask_id, dtype=np.int64)
    positive = np.asarray(bank.positive_observation, dtype=bool)
    surfels = np.asarray(surfels, dtype=np.int64)
    for frame_idx, frame_id in enumerate(frame_ids.tolist()):
        ids = target_mask_id[frame_idx, surfels]
        ids = ids[positive[frame_idx, surfels] & (ids > 0)]
        if ids.size == 0:
            continue
        for mask_id, count in Counter(int(v) for v in ids.tolist()).most_common(2):
            key = (int(frame_id), int(mask_id))
            votes[key] = max(votes.get(key, 0.0), float(count))
    src_frames = np.asarray(bank.src_frame_global[surfels], dtype=np.int64)
    src_masks = np.asarray(bank.src_mask_id[surfels], dtype=np.int64)
    for (frame_id, mask_id), count in Counter(zip(src_frames.tolist(), src_masks.tolist())).items():
        if int(mask_id) > 0:
            key = (int(frame_id), int(mask_id))
            votes[key] = max(votes.get(key, 0.0), float(count))
    out = [(frame, mask, score) for (frame, mask), score in votes.items()]
    out.sort(key=lambda item: (-float(item[2]), int(item[0]), int(item[1])))
    return out[: int(max_votes)]


def _component_labels(num_nodes: int, edges: list[tuple[int, int]]) -> list[np.ndarray]:
    adj: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen = np.zeros((int(num_nodes),), dtype=bool)
    comps: list[np.ndarray] = []
    for start in range(int(num_nodes)):
        if seen[start]:
            continue
        seen[start] = True
        q: deque[int] = deque([start])
        nodes: list[int] = []
        while q:
            node = q.popleft()
            nodes.append(int(node))
            for nxt in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        comps.append(np.asarray(nodes, dtype=np.int64))
    return comps


def _one_hop_fringe(
    graph: SignedSurfelGraph,
    evidence: SignedBoundaryEvidence,
    components: list[np.ndarray],
    *,
    max_cut_score: float,
    max_fringe_ratio: float,
) -> list[np.ndarray]:
    node_to_component: dict[int, int] = {}
    for comp_idx, comp in enumerate(components):
        for node in comp.tolist():
            node_to_component[int(node)] = int(comp_idx)
    fringe_sets: list[set[int]] = [set() for _ in components]
    core_sets = [set(int(v) for v in comp.tolist()) for comp in components]
    for a, b, score in zip(graph.src.tolist(), graph.dst.tolist(), evidence.cut_score.tolist()):
        if float(score) > float(max_cut_score):
            continue
        ca = node_to_component.get(int(a))
        cb = node_to_component.get(int(b))
        if ca is not None and cb is None:
            fringe_sets[ca].add(int(b))
        elif cb is not None and ca is None:
            fringe_sets[cb].add(int(a))
    out: list[np.ndarray] = []
    for core, fringe in zip(core_sets, fringe_sets):
        values = sorted(fringe.difference(core))
        max_keep = int(np.ceil(len(core) * float(max_fringe_ratio)))
        if max_keep > 0:
            values = values[:max_keep]
        else:
            values = []
        out.append(np.asarray(values, dtype=np.int64))
    return out


def partition_signed_graph(
    graph: SignedSurfelGraph,
    evidence: SignedBoundaryEvidence,
    *,
    mode: str = "P2_agglomerative_signed",
    cut_threshold: float = 0.62,
    merge_threshold: float = 0.55,
    min_component_size: int = 12,
    max_component_ratio: float = 0.40,
    max_fringe_ratio: float = 0.35,
    use_graph_precut: bool = True,
) -> SignedPartitionResult:
    if evidence.num_edges != graph.num_edges:
        raise ValueError("Graph/evidence edge count mismatch")
    keep_edges: list[tuple[int, int]] = []
    cut = np.asarray(evidence.cut_score, dtype=np.float32)
    merge = np.asarray(evidence.merge_weight, dtype=np.float32)
    if mode not in {"P1_signed_watershed", "P2_agglomerative_signed", "P3_seeded_graph_partition"}:
        raise ValueError(f"Unsupported partition mode: {mode}")
    precut_keep = np.asarray(getattr(graph, "precut_keep", np.ones((graph.num_edges,), dtype=bool)), dtype=bool)
    for edge_idx, (a, b, c, m) in enumerate(zip(graph.src.tolist(), graph.dst.tolist(), cut.tolist(), merge.tolist())):
        if bool(use_graph_precut) and not bool(precut_keep[edge_idx]):
            continue
        if mode == "P1_signed_watershed":
            keep = float(c) < float(cut_threshold)
        elif mode == "P2_agglomerative_signed":
            keep = float(c) < float(cut_threshold) and (float(m) >= float(merge_threshold) or float(c) < 0.45)
        else:
            # Seeded v1 is conservative: only very low cut edges are used.
            keep = float(c) < min(float(cut_threshold), 0.50)
        if keep:
            keep_edges.append((int(a), int(b)))
    raw_components = _component_labels(graph.num_nodes, keep_edges)
    max_size = int(np.ceil(graph.num_nodes * float(max_component_ratio)))
    components = [
        comp
        for comp in raw_components
        if int(comp.shape[0]) >= int(min_component_size) and int(comp.shape[0]) <= max(max_size, int(min_component_size))
    ]
    components.sort(key=lambda arr: (-int(arr.shape[0]), int(arr[0]) if arr.size else -1))
    fringe = _one_hop_fringe(
        graph,
        evidence,
        components,
        max_cut_score=min(float(cut_threshold), 0.55),
        max_fringe_ratio=float(max_fringe_ratio),
    )
    sizes = np.asarray([comp.shape[0] for comp in components], dtype=np.float64)
    diagnostics = {
        "scene": graph.scene,
        "mode": mode,
        "num_raw_components": int(len(raw_components)),
        "num_kept_components": int(len(components)),
        "mean_component_size": float(np.mean(sizes)) if sizes.size else 0.0,
        "median_component_size": float(np.median(sizes)) if sizes.size else 0.0,
        "largest_component_ratio": float(np.max(sizes) / max(graph.num_nodes, 1)) if sizes.size else 0.0,
        "tiny_component_ratio": float(
            sum(1 for comp in raw_components if comp.shape[0] < int(min_component_size)) / max(len(raw_components), 1)
        ),
        "cut_threshold": float(cut_threshold),
        "merge_threshold": float(merge_threshold),
        "min_component_size": int(min_component_size),
        "max_component_ratio": float(max_component_ratio),
        "max_fringe_ratio": float(max_fringe_ratio),
        "use_graph_precut": bool(use_graph_precut),
        "graph_precut_removed_edge_ratio": float(1.0 - np.count_nonzero(precut_keep) / max(graph.num_edges, 1)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
    }
    return SignedPartitionResult(scene=graph.scene, mode=mode, components=components, fringe_components=fringe, diagnostics=diagnostics)


def partition_to_object_dict(
    bank: MeasurementBank,
    result: SignedPartitionResult,
    *,
    export_mode: str,
    max_mask_votes: int = 8,
) -> dict[int, dict[str, Any]]:
    if export_mode not in {"G_core", "G_region_fill"}:
        raise ValueError(f"Unsupported v18 export mode: {export_mode}")
    out: dict[int, dict[str, Any]] = {}
    for idx, comp in enumerate(result.components):
        fringe = result.fringe_components[idx] if export_mode == "G_region_fill" else np.empty((0,), dtype=np.int64)
        all_for_votes = np.concatenate([comp, fringe]) if fringe.size else comp
        mask_list = _mask_votes(bank, all_for_votes, max_votes=int(max_mask_votes))
        if not mask_list:
            continue
        out[len(out)] = {
            "mask_list": mask_list,
            "carrier_ids": comp.astype(np.int64, copy=False),
            "core_surfels": comp.astype(np.int64, copy=False),
            "fringe_surfels": fringe.astype(np.int64, copy=False),
            "unknown_surfels": np.empty((0,), dtype=np.int64),
            "reject_surfels": np.empty((0,), dtype=np.int64),
            "v18_partition": {
                "mode": result.mode,
                "export_mode": export_mode,
                "component_size": int(comp.shape[0]),
                "fringe_size": int(fringe.shape[0]),
            },
        }
    return out


def component_gt_diagnostics(components: list[np.ndarray], surfel_gt: np.ndarray) -> dict[str, Any]:
    valid_gt = np.asarray(surfel_gt, dtype=np.int64)
    purities: list[float] = []
    gt_to_components: dict[int, int] = defaultdict(int)
    for comp in components:
        labels = valid_gt[np.asarray(comp, dtype=np.int64)]
        labels = labels[labels >= 1000]
        if labels.size == 0:
            continue
        values, counts = np.unique(labels, return_counts=True)
        purities.append(float(np.max(counts) / max(labels.shape[0], 1)))
        gt_to_components[int(values[int(np.argmax(counts))])] += 1
    return {
        "component_purity_mean": float(np.mean(purities)) if purities else None,
        "component_purity_p10": float(np.percentile(purities, 10)) if purities else None,
        "oversegmentation_per_gt_mean": float(np.mean(list(gt_to_components.values()))) if gt_to_components else None,
        "gt_instances_touched_by_components": int(len(gt_to_components)),
    }
