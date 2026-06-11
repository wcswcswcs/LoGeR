from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


@dataclass
class TypedNode:
    node_id: int
    frame_id: int
    mask_id: int
    coverage: float
    point_ids: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    observation_hits: int

    @property
    def is_weak(self) -> bool:
        return bool(getattr(self, "weak", False))


@dataclass
class TypedEdge:
    edge_type: str
    left: int
    right: int
    score: float
    shared_points: int
    reason: str = ""


class TypedComponentIndex:
    def __init__(self, nodes: list[TypedNode], cannot_links: set[tuple[int, int]]) -> None:
        self.nodes = nodes
        self.parent = list(range(len(nodes)))
        self.members: dict[int, set[int]] = {idx: {idx} for idx in range(len(nodes))}
        self.frame_masks: dict[int, dict[int, int]] = {
            idx: {int(node.frame_id): int(node.mask_id)} for idx, node in enumerate(nodes)
        }
        self.strong_count: dict[int, int] = {
            idx: (0 if bool(getattr(node, "weak", False)) else 1) for idx, node in enumerate(nodes)
        }
        self.cannot_links = {tuple(sorted((int(a), int(b)))) for a, b in cannot_links}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def _violates_cannot_link(self, left_root: int, right_root: int) -> bool:
        for left in self.members[left_root]:
            for right in self.members[right_root]:
                if tuple(sorted((int(left), int(right)))) in self.cannot_links:
                    return True
        return False

    def can_union(self, left: int, right: int) -> tuple[bool, str]:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False, "same_component"
        for frame_id, mask_id in self.frame_masks[root_left].items():
            other = self.frame_masks[root_right].get(frame_id)
            if other is not None and int(other) != int(mask_id):
                return False, "same_frame_conflict"
        if self._violates_cannot_link(root_left, root_right):
            return False, "cannot_link"
        return True, ""

    def can_weak_attach(self, left: int, right: int) -> tuple[bool, str]:
        ok, reason = self.can_union(left, right)
        if not ok:
            return False, reason
        root_left = self.find(left)
        root_right = self.find(right)
        left_strong = int(self.strong_count[root_left])
        right_strong = int(self.strong_count[root_right])
        if (left_strong == 0 and right_strong > 0) or (right_strong == 0 and left_strong > 0):
            return True, ""
        return False, "weak_bridge_between_strong_components"

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        ok, _ = self.can_union(root_left, root_right)
        if not ok:
            return False
        if len(self.members[root_left]) < len(self.members[root_right]):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.members[root_left].update(self.members.pop(root_right))
        self.frame_masks[root_left].update(self.frame_masks.pop(root_right))
        self.strong_count[root_left] += self.strong_count.pop(root_right)
        return True

    def components(self) -> list[list[int]]:
        return [sorted(self.members[root]) for root in sorted(self.members)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_observation_index(debug_root: Path, seq_name: str, min_coverage: float) -> list[dict[str, Any]]:
    seq_dir = debug_root / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(seq_dir)
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for window_path in sorted(seq_dir.glob("local_props_window*.json")):
        with window_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for prop in payload.get("proposals", []):
            for item in prop.get("mask_observations", []):
                coverage = float(item.get("coverage", 0.0))
                if coverage < float(min_coverage):
                    continue
                key = (int(item["frame_id"]), int(item["mask_id"]))
                record = by_key.setdefault(
                    key,
                    {
                        "frame_id": key[0],
                        "mask_id": key[1],
                        "coverage": 0.0,
                        "observation_hits": 0,
                        "windows": set(),
                    },
                )
                record["coverage"] = max(float(record["coverage"]), coverage)
                record["observation_hits"] = int(record["observation_hits"]) + 1
                record["windows"].add(window_path.stem)
    observations = list(by_key.values())
    observations.sort(key=lambda item: (int(item["frame_id"]), int(item["mask_id"])))
    for item in observations:
        item["windows"] = sorted(item["windows"])
    return observations


def _points_summary(scene_points: np.ndarray, point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = scene_points[np.asarray(point_ids, dtype=np.int64)]
    return pts.mean(axis=0), pts.min(axis=0), pts.max(axis=0)


def _backproject_nodes(
    exporter: ScanNetExporter,
    observations: list[dict[str, Any]],
    min_points: int,
    nn_radius: float,
    weak_coverage: float,
) -> tuple[list[TypedNode], dict[str, float]]:
    nodes: list[TypedNode] = []
    total_queries = 0
    total_hits = 0
    dropped_small = 0
    for item in observations:
        point_ids, query_count = exporter._backproject_mask(  # diagnostic method-building tool.
            int(item["frame_id"]),
            int(item["mask_id"]),
            nn_radius=float(nn_radius),
        )
        total_queries += int(query_count)
        total_hits += int(point_ids.shape[0])
        point_ids = np.unique(point_ids.astype(np.int64))
        if point_ids.shape[0] < int(min_points):
            dropped_small += 1
            continue
        centroid, bbox_min, bbox_max = _points_summary(exporter.scene_points, point_ids)
        node = TypedNode(
            node_id=len(nodes),
            frame_id=int(item["frame_id"]),
            mask_id=int(item["mask_id"]),
            coverage=float(item["coverage"]),
            point_ids=point_ids,
            centroid=centroid.astype(np.float32),
            bbox_min=bbox_min.astype(np.float32),
            bbox_max=bbox_max.astype(np.float32),
            observation_hits=int(item["observation_hits"]),
        )
        setattr(node, "weak", bool(float(item["coverage"]) <= float(weak_coverage)))
        nodes.append(node)
    return nodes, {
        "typed_v3_loaded_observations": float(len(observations)),
        "typed_v3_nodes": float(len(nodes)),
        "typed_v3_dropped_small_nodes": float(dropped_small),
        "typed_v3_backproject_queries": float(total_queries),
        "typed_v3_backproject_hits": float(total_hits),
        "typed_v3_backproject_hit_rate": float(total_hits / max(total_queries, 1)),
    }


def _bbox_gap(left: TypedNode, right: TypedNode) -> float:
    gap = np.maximum(0.0, np.maximum(left.bbox_min - right.bbox_max, right.bbox_min - left.bbox_max))
    return float(np.linalg.norm(gap))


def _edge_candidates(
    nodes: list[TypedNode],
    *,
    min_track_shared: int,
    min_track_ioc: float,
    min_weak_shared: int,
    min_weak_ioc: float,
    min_conflict_shared: int,
    min_conflict_ioc: float,
    complement_max_centroid: float,
    complement_max_bbox_gap: float,
) -> tuple[list[TypedEdge], set[tuple[int, int]], dict[str, float]]:
    point_to_nodes: dict[int, list[int]] = {}
    for idx, node in enumerate(nodes):
        for point_id in node.point_ids.tolist():
            point_to_nodes.setdefault(int(point_id), []).append(idx)
    intersections: Counter[tuple[int, int]] = Counter()
    for owners in point_to_nodes.values():
        if len(owners) < 2:
            continue
        owners = sorted(set(owners))
        for pos, left in enumerate(owners):
            for right in owners[pos + 1 :]:
                intersections[(left, right)] += 1

    edges: list[TypedEdge] = []
    cannot_links: set[tuple[int, int]] = set()
    same_frame_pairs = 0
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            shared = int(intersections.get((left, right), 0))
            denom = max(1, min(int(nodes[left].point_ids.shape[0]), int(nodes[right].point_ids.shape[0])))
            ioc = float(shared / denom)
            same_frame = int(nodes[left].frame_id) == int(nodes[right].frame_id)
            if same_frame:
                same_frame_pairs += 1
                if shared >= int(min_conflict_shared) and ioc >= float(min_conflict_ioc):
                    cannot_links.add((left, right))
                    edges.append(TypedEdge("negative_conflict", left, right, ioc, shared, "same_frame_overlap"))
                continue
            if shared >= int(min_track_shared) and ioc >= float(min_track_ioc):
                if bool(getattr(nodes[left], "weak", False)) or bool(getattr(nodes[right], "weak", False)):
                    edges.append(TypedEdge("weak_bridge", left, right, ioc, shared, "track_overlap_with_weak_endpoint"))
                else:
                    edges.append(TypedEdge("positive_track", left, right, ioc, shared, "carrier_backproject_overlap"))
                continue
            if shared >= int(min_weak_shared) and ioc >= float(min_weak_ioc):
                edges.append(TypedEdge("weak_bridge", left, right, ioc, shared, "low_overlap_bridge"))
                continue
            if complement_max_centroid > 0.0:
                centroid_dist = float(np.linalg.norm(nodes[left].centroid - nodes[right].centroid))
                bbox_gap = _bbox_gap(nodes[left], nodes[right])
                if centroid_dist <= float(complement_max_centroid) and bbox_gap <= float(complement_max_bbox_gap):
                    if not bool(getattr(nodes[left], "weak", False)) and not bool(getattr(nodes[right], "weak", False)):
                        edges.append(
                            TypedEdge(
                                "positive_complement",
                                left,
                                right,
                                1.0 / (1.0 + centroid_dist),
                                shared,
                                "geometry_compatible",
                            )
                        )
                    else:
                        edges.append(TypedEdge("weak_bridge", left, right, 1.0 / (1.0 + centroid_dist), shared, "weak_geometry_bridge"))
    type_counts = Counter(edge.edge_type for edge in edges)
    diagnostics = {
        "typed_v3_same_frame_pairs": float(same_frame_pairs),
        "typed_v3_cannot_link_pairs": float(len(cannot_links)),
        "typed_v3_edge_candidates": float(len(edges)),
        "typed_v3_positive_track_edges": float(type_counts.get("positive_track", 0)),
        "typed_v3_positive_complement_edges": float(type_counts.get("positive_complement", 0)),
        "typed_v3_weak_bridge_edges": float(type_counts.get("weak_bridge", 0)),
        "typed_v3_negative_conflict_edges": float(type_counts.get("negative_conflict", 0)),
    }
    return edges, cannot_links, diagnostics


def _build_components(nodes: list[TypedNode], edges: list[TypedEdge], cannot_links: set[tuple[int, int]]) -> tuple[list[list[int]], dict[str, float]]:
    index = TypedComponentIndex(nodes, cannot_links)
    accepted = Counter()
    rejected = Counter()
    positive_edges = [edge for edge in edges if edge.edge_type in {"positive_track", "positive_complement"}]
    weak_edges = [edge for edge in edges if edge.edge_type == "weak_bridge"]
    positive_edges.sort(key=lambda item: (item.edge_type == "positive_track", item.score, item.shared_points), reverse=True)
    weak_edges.sort(key=lambda item: (item.score, item.shared_points), reverse=True)
    for edge in positive_edges:
        ok, reason = index.can_union(edge.left, edge.right)
        if not ok:
            rejected[f"{edge.edge_type}:{reason}"] += 1
            continue
        if index.union(edge.left, edge.right):
            accepted[edge.edge_type] += 1
    for edge in weak_edges:
        ok, reason = index.can_weak_attach(edge.left, edge.right)
        if not ok:
            rejected[f"{edge.edge_type}:{reason}"] += 1
            continue
        if index.union(edge.left, edge.right):
            accepted[edge.edge_type] += 1
    diagnostics: dict[str, float] = {
        "typed_v3_accepted_positive_track": float(accepted.get("positive_track", 0)),
        "typed_v3_accepted_positive_complement": float(accepted.get("positive_complement", 0)),
        "typed_v3_accepted_weak_bridge": float(accepted.get("weak_bridge", 0)),
        "typed_v3_rejected_edges_total": float(sum(rejected.values())),
    }
    for key, value in sorted(rejected.items()):
        diagnostics[f"typed_v3_rejected_{key.replace(':', '_')}"] = float(value)
    return index.components(), diagnostics


def _component_score(nodes: list[TypedNode], member_ids: list[int], mode: str) -> float:
    coverages = np.asarray([nodes[idx].coverage for idx in member_ids], dtype=np.float64)
    frames = {int(nodes[idx].frame_id) for idx in member_ids}
    points = set().union(*(set(nodes[idx].point_ids.tolist()) for idx in member_ids))
    strong_nodes = [idx for idx in member_ids if not bool(getattr(nodes[idx], "weak", False))]
    if mode == "quality":
        return float(len(strong_nodes) * np.sqrt(max(len(points), 1)) * max(float(np.mean(coverages)), 1e-6) * np.sqrt(max(len(frames), 1)))
    if mode == "observations":
        return float(len(member_ids))
    if mode == "points":
        return float(len(points))
    if mode == "mean_coverage":
        return float(np.mean(coverages))
    raise ValueError(f"Unsupported score mode: {mode}")


def _component_support_points(
    exporter: ScanNetExporter,
    nodes: list[TypedNode],
    member_ids: list[int],
    *,
    support_mode: str,
    min_core_frames: int,
    min_core_observations: int,
    fringe_radius: float,
) -> tuple[list[int], dict[str, int]]:
    point_frames: dict[int, set[int]] = {}
    point_observations: Counter[int] = Counter()
    for idx in member_ids:
        frame_id = int(nodes[idx].frame_id)
        for point_id in nodes[idx].point_ids.tolist():
            point_key = int(point_id)
            point_frames.setdefault(point_key, set()).add(frame_id)
            point_observations[point_key] += 1

    union_ids = sorted(point_frames)
    core_ids = [
        point_id
        for point_id in union_ids
        if len(point_frames[point_id]) >= int(min_core_frames)
        and int(point_observations[point_id]) >= int(min_core_observations)
    ]
    core_set = set(core_ids)
    fringe_ids = [point_id for point_id in union_ids if point_id not in core_set]

    if support_mode == "union":
        selected_ids = union_ids
        fringe_kept = len(fringe_ids)
    elif support_mode == "core":
        selected_ids = core_ids
        fringe_kept = 0
    elif support_mode == "core_connected_fringe":
        if not core_ids or not fringe_ids:
            selected_ids = core_ids
            fringe_kept = 0
        else:
            core_points = exporter.scene_points[np.asarray(core_ids, dtype=np.int64)]
            fringe_points = exporter.scene_points[np.asarray(fringe_ids, dtype=np.int64)]
            tree = cKDTree(core_points)
            distance, _ = tree.query(fringe_points, k=1, workers=-1)
            keep_fringe = [point_id for point_id, dist in zip(fringe_ids, distance.tolist()) if float(dist) <= float(fringe_radius)]
            selected_ids = sorted(core_ids + keep_fringe)
            fringe_kept = len(keep_fringe)
    else:
        raise ValueError(f"Unsupported support mode: {support_mode}")

    return selected_ids, {
        "union_points": int(len(union_ids)),
        "core_points": int(len(core_ids)),
        "fringe_candidate_points": int(len(fringe_ids)),
        "fringe_kept_points": int(fringe_kept),
        "reject_points": int(max(len(union_ids) - len(selected_ids), 0)),
    }


def _write_outputs(
    *,
    exporter: ScanNetExporter,
    nodes: list[TypedNode],
    components: list[list[int]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pred_masks: list[np.ndarray] = []
    pred_scores: list[float] = []
    object_dict: dict[int, dict[str, Any]] = {}
    kept_components: list[dict[str, Any]] = []
    dropped = 0
    support_stats: list[dict[str, int]] = []
    for component in components:
        frames = {int(nodes[idx].frame_id) for idx in component}
        if len(component) < int(args.min_component_observations) or len(frames) < int(args.min_component_frames):
            dropped += 1
            continue
        point_ids, component_support = _component_support_points(
            exporter,
            nodes,
            component,
            support_mode=args.support_mode,
            min_core_frames=int(args.min_core_frames),
            min_core_observations=int(args.min_core_observations),
            fringe_radius=float(args.fringe_radius),
        )
        if len(point_ids) < int(args.min_points_per_object):
            dropped += 1
            continue
        support_stats.append(component_support)
        out_id = len(pred_masks)
        mask = np.zeros((exporter.scene_points.shape[0],), dtype=bool)
        mask[np.asarray(point_ids, dtype=np.int64)] = True
        pred_masks.append(mask)
        score = _component_score(nodes, component, args.score_mode)
        pred_scores.append(score)
        mask_list = [(int(nodes[idx].frame_id), int(nodes[idx].mask_id), float(nodes[idx].coverage)) for idx in component]
        object_dict[out_id] = {
            "point_ids": np.asarray(point_ids, dtype=np.int64),
            "mask_list": sorted(mask_list, key=lambda item: (item[0], item[1])),
            "repre_mask_list": sorted(mask_list, key=lambda item: item[2], reverse=True)[:5],
            "carrier_ids": np.empty((0,), dtype=np.int64),
        }
        kept_components.append(
            {
                "object_id": int(out_id),
                "num_nodes": int(len(component)),
                "num_frames": int(len(frames)),
                "num_points": int(len(point_ids)),
                "num_weak_nodes": int(sum(1 for idx in component if bool(getattr(nodes[idx], "weak", False)))),
                **component_support,
                "score": float(score),
            }
        )

    if pred_masks:
        pred_mask_np = np.stack(pred_masks, axis=1).astype(bool, copy=False)
        pred_score_np = np.asarray(pred_scores, dtype=np.float32)
    else:
        pred_mask_np = np.zeros((exporter.scene_points.shape[0], 0), dtype=bool)
        pred_score_np = np.zeros((0,), dtype=np.float32)
    pred_classes = np.zeros((pred_score_np.shape[0],), dtype=np.int32)
    pred_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{args.seq_name}.npz",
        pred_masks=pred_mask_np,
        pred_score=pred_score_np,
        pred_classes=pred_classes,
    )
    tmp_dir = Path("data/TMP") / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_points = np.flatnonzero(pred_mask_np.any(axis=1)).astype(np.int64)
    np.save(tmp_dir / f"{args.seq_name}_pre_points.npy", pre_points)
    object_dir = exporter.stream.object_dir / args.output_config
    object_dir.mkdir(parents=True, exist_ok=True)
    np.save(object_dir / "object_dict.npy", object_dict, allow_pickle=True)
    support_totals = Counter()
    for item in support_stats:
        support_totals.update({key: int(value) for key, value in item.items()})
    denom = max(int(support_totals.get("union_points", 0)), 1)
    return {
        "kept_components": int(len(pred_scores)),
        "dropped_components": int(dropped),
        "union_points": int(pre_points.shape[0]),
        "support_mode": args.support_mode,
        "component_union_points_total": int(support_totals.get("union_points", 0)),
        "core_points_total": int(support_totals.get("core_points", 0)),
        "fringe_candidate_points_total": int(support_totals.get("fringe_candidate_points", 0)),
        "fringe_kept_points_total": int(support_totals.get("fringe_kept_points", 0)),
        "reject_points_total": int(support_totals.get("reject_points", 0)),
        "core_ratio_total": float(support_totals.get("core_points", 0) / denom),
        "fringe_kept_ratio_total": float(support_totals.get("fringe_kept_points", 0) / denom),
        "reject_ratio_total": float(support_totals.get("reject_points", 0) / denom),
        "score_min": float(np.min(pred_score_np)) if pred_score_np.size else 0.0,
        "score_mean": float(np.mean(pred_score_np)) if pred_score_np.size else 0.0,
        "score_max": float(np.max(pred_score_np)) if pred_score_np.size else 0.0,
        "components_preview": kept_components[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a minimal typed evidence graph v3 prediction from cached Stream4D local props.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-coverage", type=float, default=0.001)
    parser.add_argument("--weak-coverage", type=float, default=0.004)
    parser.add_argument("--min-points-per-mask", type=int, default=80)
    parser.add_argument("--min-points-per-object", type=int, default=100)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--min-track-shared", type=int, default=30)
    parser.add_argument("--min-track-ioc", type=float, default=0.35)
    parser.add_argument("--min-weak-shared", type=int, default=20)
    parser.add_argument("--min-weak-ioc", type=float, default=0.20)
    parser.add_argument("--min-conflict-shared", type=int, default=25)
    parser.add_argument("--min-conflict-ioc", type=float, default=0.15)
    parser.add_argument("--complement-max-centroid", type=float, default=0.20)
    parser.add_argument("--complement-max-bbox-gap", type=float, default=0.05)
    parser.add_argument("--min-component-observations", type=int, default=2)
    parser.add_argument("--min-component-frames", type=int, default=2)
    parser.add_argument("--support-mode", default="union", choices=["union", "core", "core_connected_fringe"])
    parser.add_argument("--min-core-frames", type=int, default=2)
    parser.add_argument("--min-core-observations", type=int, default=2)
    parser.add_argument("--fringe-radius", type=float, default=0.08)
    parser.add_argument("--score-mode", default="quality", choices=["quality", "observations", "points", "mean_coverage"])
    parser.add_argument("--summary-root", default="outputs/typed_evidence_graph_v6")
    args = parser.parse_args()

    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(stream, output_config=args.output_config, export_nn_radius=args.export_nn_radius)
    observations = _load_observation_index(Path(args.debug_root), args.seq_name, min_coverage=float(args.min_coverage))
    nodes, node_diag = _backproject_nodes(
        exporter,
        observations,
        min_points=int(args.min_points_per_mask),
        nn_radius=float(args.export_nn_radius),
        weak_coverage=float(args.weak_coverage),
    )
    edges, cannot_links, edge_diag = _edge_candidates(
        nodes,
        min_track_shared=int(args.min_track_shared),
        min_track_ioc=float(args.min_track_ioc),
        min_weak_shared=int(args.min_weak_shared),
        min_weak_ioc=float(args.min_weak_ioc),
        min_conflict_shared=int(args.min_conflict_shared),
        min_conflict_ioc=float(args.min_conflict_ioc),
        complement_max_centroid=float(args.complement_max_centroid),
        complement_max_bbox_gap=float(args.complement_max_bbox_gap),
    )
    components, union_diag = _build_components(nodes, edges, cannot_links)
    export_diag = _write_outputs(exporter=exporter, nodes=nodes, components=components, args=args)
    weak_nodes = int(sum(1 for node in nodes if bool(getattr(node, "weak", False))))
    summary = {
        "args": vars(args),
        "typed_v3_algorithm": "positive_track + positive_complement + negative_conflict + weak_bridge_attach_only",
        "typed_v3_raw_components": int(len(components)),
        "typed_v3_weak_nodes": int(weak_nodes),
        "typed_v3_strong_nodes": int(len(nodes) - weak_nodes),
        **node_diag,
        **edge_diag,
        **union_diag,
        **export_diag,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.output_config}_{args.seq_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False, sort_keys=True)
    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.debug_root)],
        pre_points_policy="recompute",
        support_policy="typed_evidence_graph_v3_mask_backproject",
        notes="Minimal v6 typed evidence graph candidate from cached Stream4D local props; no GT used.",
        extra={
            "typed_v3_algorithm": summary["typed_v3_algorithm"],
            "typed_v3_summary_root": str(out_dir),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".", pred_suffix="class_agnostic")
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
