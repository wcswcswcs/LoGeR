from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.reliable_densifier import ReliableDensifier, ReliableDensifyParams, apply_wta_to_records
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _local_to_global(src_frame: np.ndarray, src_frame_global: np.ndarray, num_frames: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for local in range(int(num_frames)):
        hits = np.flatnonzero(src_frame == int(local))
        if hits.size:
            out[int(local)] = int(src_frame_global[int(hits[0])])
    return out


def _load_mask_resized(stream: ScanNetStream, frame_id: int) -> np.ndarray:
    mask = stream.load_mask(int(frame_id))
    depth = stream.load_depth(int(frame_id))
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def _backproject_xy_with_index(
    stream: ScanNetStream,
    tree,
    scene_points: np.ndarray,
    intrinsics: np.ndarray,
    frame_id: int,
    xy: np.ndarray,
    nn_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.int64)
    if xy.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    depth = stream.load_depth(int(frame_id))
    pose = stream.load_pose(int(frame_id))
    if not np.isfinite(pose).all():
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    h, w = depth.shape
    x = xy[:, 0]
    y = xy[:, 1]
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(in_bounds):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    input_indices = np.flatnonzero(in_bounds)
    x = x[in_bounds]
    y = y[in_bounds]
    z = depth[y, x]
    depth_valid = np.isfinite(z) & (z > 0.0)
    if not np.any(depth_valid):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    input_indices = input_indices[depth_valid]
    x = x[depth_valid].astype(np.float32)
    y = y[depth_valid].astype(np.float32)
    z = z[depth_valid].astype(np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    cam = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z, np.ones_like(z)], axis=1)
    world = (pose @ cam.T).T[:, :3].astype(np.float32)
    finite = np.isfinite(world).all(axis=1)
    if not np.any(finite):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    input_indices = input_indices[finite]
    dist, point_ids = tree.query(world[finite], k=1, distance_upper_bound=float(nn_radius))
    hit = np.isfinite(dist) & (point_ids < scene_points.shape[0])
    return input_indices[hit].astype(np.int64), point_ids[hit].astype(np.int64)


class CarrierComponentIndex:
    def __init__(self, memberships: list[dict[int, int]], max_masks_per_frame: int, max_component_carriers: int) -> None:
        self.parent = list(range(len(memberships)))
        self.members = {idx: {idx} for idx in range(len(memberships))}
        self.frame_masks: dict[int, dict[int, set[int]]] = {
            idx: {frame_id: {mask_id} for frame_id, mask_id in item.items()}
            for idx, item in enumerate(memberships)
        }
        self.max_masks_per_frame = max(1, int(max_masks_per_frame))
        self.max_component_carriers = int(max_component_carriers)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def can_union(self, left: int, right: int) -> tuple[bool, str]:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False, "same_component"
        if self.max_component_carriers > 0 and (
            len(self.members[root_left]) + len(self.members[root_right]) > self.max_component_carriers
        ):
            return False, "component_size"
        left_masks = self.frame_masks[root_left]
        right_masks = self.frame_masks[root_right]
        for frame_id in set(left_masks).intersection(right_masks):
            merged = left_masks[frame_id] | right_masks[frame_id]
            if len(merged) > self.max_masks_per_frame:
                return False, "same_frame_cannot_link"
        return True, ""

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
        for frame_id, masks in self.frame_masks.pop(root_right).items():
            self.frame_masks[root_left].setdefault(frame_id, set()).update(masks)
        return True

    def components(self) -> list[list[int]]:
        return [sorted(items) for _, items in sorted(self.members.items())]


def _window_memberships(
    stream: ScanNetStream,
    carrier_path: Path,
    *,
    min_visibility: float,
    min_confidence: float,
) -> dict[str, Any]:
    with np.load(carrier_path) as data:
        carrier_id = np.asarray(data["carrier_id"], dtype=np.int64)
        src_frame = np.asarray(data["src_frame"], dtype=np.int64)
        src_frame_global = np.asarray(data.get("src_frame_global", src_frame), dtype=np.int64)
        src_xy = np.asarray(data["src_xy"], dtype=np.int64)
        src_mask_id = np.asarray(data.get("src_mask_id", np.zeros_like(src_frame)), dtype=np.int64)
        uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
        valid = np.asarray(data.get("valid", np.ones(uv_pred.shape[:2], dtype=bool)), dtype=bool)
        visibility = np.asarray(data.get("visibility_prob", np.ones(uv_pred.shape[:2], dtype=np.float32)), dtype=np.float32)
        confidence = np.asarray(data.get("confidence_prob", np.ones(uv_pred.shape[:2], dtype=np.float32)), dtype=np.float32)
        xyz_ref = np.asarray(data.get("xyz_ref", np.empty((0, 0, 3), dtype=np.float32)), dtype=np.float32)

    num_frames, num_carriers = int(uv_pred.shape[0]), int(uv_pred.shape[1])
    local_to_global = _local_to_global(src_frame, src_frame_global, num_frames)
    memberships: list[dict[int, int]] = [dict() for _ in range(num_carriers)]
    mask_to_carriers: dict[tuple[int, int], list[int]] = {}
    valid_assignments = 0

    for local_frame in range(num_frames):
        frame_id = local_to_global.get(local_frame)
        if frame_id is None:
            continue
        mask = _load_mask_resized(stream, frame_id)
        h, w = mask.shape
        uv = uv_pred[local_frame]
        ok = (
            valid[local_frame]
            & (visibility[local_frame] >= float(min_visibility))
            & (confidence[local_frame] >= float(min_confidence))
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
        )
        if not np.any(ok):
            continue
        carrier_indices = np.flatnonzero(ok)
        x = np.rint(uv[carrier_indices, 0] * float(max(w - 1, 1))).astype(np.int64)
        y = np.rint(uv[carrier_indices, 1] * float(max(h - 1, 1))).astype(np.int64)
        mask_ids = mask[y, x].astype(np.int64)
        keep = mask_ids > 0
        for carrier_idx, mask_id in zip(carrier_indices[keep].tolist(), mask_ids[keep].tolist()):
            memberships[int(carrier_idx)][int(frame_id)] = int(mask_id)
            mask_to_carriers.setdefault((int(frame_id), int(mask_id)), []).append(int(carrier_idx))
            valid_assignments += 1

    for carrier_idx, (frame_id, mask_id) in enumerate(zip(src_frame_global.tolist(), src_mask_id.tolist())):
        if int(mask_id) <= 0:
            continue
        memberships[int(carrier_idx)].setdefault(int(frame_id), int(mask_id))
        mask_to_carriers.setdefault((int(frame_id), int(mask_id)), []).append(int(carrier_idx))

    return {
        "carrier_path": str(carrier_path),
        "carrier_id": carrier_id,
        "src_frame_global": src_frame_global,
        "src_xy": src_xy,
        "xyz_ref": xyz_ref,
        "memberships": memberships,
        "mask_to_carriers": mask_to_carriers,
        "valid_assignments": valid_assignments,
        "num_carriers": num_carriers,
    }


def _trajectory_variance(xyz_ref: np.ndarray, left: int, right: int) -> float:
    if xyz_ref.ndim != 3 or xyz_ref.shape[0] == 0:
        return 0.0
    left_pts = xyz_ref[:, int(left), :]
    right_pts = xyz_ref[:, int(right), :]
    ok = np.isfinite(left_pts).all(axis=1) & np.isfinite(right_pts).all(axis=1)
    if np.count_nonzero(ok) < 2:
        return 0.0
    distances = np.linalg.norm(left_pts[ok] - right_pts[ok], axis=1)
    return float(np.var(distances))


def _build_edges(
    window: dict[str, Any],
    *,
    min_shared_frames: int,
    min_positive_ratio: float,
    max_carriers_per_mask_edge: int,
    max_pair_distance_variance: float,
) -> tuple[list[tuple[int, int, float, int]], dict[str, float]]:
    shared: Counter[tuple[int, int]] = Counter()
    for carriers in window["mask_to_carriers"].values():
        carriers = sorted(set(int(v) for v in carriers))
        if len(carriers) > int(max_carriers_per_mask_edge) > 0:
            pick = np.linspace(0, len(carriers) - 1, num=int(max_carriers_per_mask_edge), dtype=np.int64)
            carriers = [carriers[int(idx)] for idx in pick.tolist()]
        for pos, left in enumerate(carriers):
            for right in carriers[pos + 1 :]:
                shared[(left, right)] += 1
    visible_counts = np.asarray([len(item) for item in window["memberships"]], dtype=np.int64)
    edges: list[tuple[int, int, float, int]] = []
    rejected_by_variance = 0
    for (left, right), count in shared.items():
        if int(count) < int(min_shared_frames):
            continue
        denom = max(1, min(int(visible_counts[left]), int(visible_counts[right])))
        score = float(count / denom)
        if score < float(min_positive_ratio):
            continue
        variance = _trajectory_variance(window["xyz_ref"], left, right)
        if float(max_pair_distance_variance) > 0.0 and variance > float(max_pair_distance_variance):
            rejected_by_variance += 1
            continue
        edges.append((int(left), int(right), score, int(count)))
    edges.sort(key=lambda item: (item[2], item[3]), reverse=True)
    return edges, {
        "positive_edge_candidates": float(len(edges)),
        "pair_candidates_with_shared_masks": float(len(shared)),
        "positive_edges_rejected_by_trajectory_variance": float(rejected_by_variance),
    }


def _cluster_window(window: dict[str, Any], args: argparse.Namespace) -> tuple[list[list[int]], dict[str, float]]:
    edges, edge_diag = _build_edges(
        window,
        min_shared_frames=int(args.min_shared_frames),
        min_positive_ratio=float(args.min_positive_ratio),
        max_carriers_per_mask_edge=int(args.max_carriers_per_mask_edge),
        max_pair_distance_variance=float(args.max_pair_distance_variance),
    )
    index = CarrierComponentIndex(
        window["memberships"],
        max_masks_per_frame=int(args.max_masks_per_frame_component),
        max_component_carriers=int(args.max_component_carriers),
    )
    accepted = 0
    rejected = Counter()
    for left, right, _, _ in edges:
        ok, reason = index.can_union(left, right)
        if not ok:
            rejected[reason] += 1
            continue
        if index.union(left, right):
            accepted += 1
    components = index.components()
    diag = {
        **edge_diag,
        "accepted_positive_edges": float(accepted),
        "rejected_unions_total": float(sum(rejected.values())),
        "raw_components": float(len(components)),
    }
    for key, value in sorted(rejected.items()):
        diag[f"rejected_unions_by_{key}"] = float(value)
    return components, diag


def _selected_masks_for_component(
    memberships: list[dict[int, int]],
    component: list[int],
    min_mask_carriers: int,
    min_frame_mask_ratio: float,
) -> list[tuple[int, int, int, float]]:
    by_frame: dict[int, Counter[int]] = {}
    for carrier_idx in component:
        for frame_id, mask_id in memberships[int(carrier_idx)].items():
            by_frame.setdefault(int(frame_id), Counter())[int(mask_id)] += 1
    out: list[tuple[int, int, int, float]] = []
    for frame_id, counts in sorted(by_frame.items()):
        total = max(sum(counts.values()), 1)
        for mask_id, count in counts.items():
            ratio = float(count / total)
            if int(count) >= int(min_mask_carriers) and ratio >= float(min_frame_mask_ratio):
                out.append((int(frame_id), int(mask_id), int(count), ratio))
    return out


def _core_points_for_component(
    stream: ScanNetStream,
    tree,
    scene_points: np.ndarray,
    intrinsics: np.ndarray,
    window: dict[str, Any],
    component: list[int],
    nn_radius: float,
) -> set[int]:
    component_set = set(int(v) for v in component)
    point_ids: set[int] = set()
    src_frames = window["src_frame_global"]
    src_xy = window["src_xy"]
    for frame_id in sorted(set(int(src_frames[idx]) for idx in component)):
        carrier_indices = [idx for idx in component if int(src_frames[idx]) == int(frame_id)]
        if not carrier_indices:
            continue
        input_indices, hits = _backproject_xy_with_index(
            stream,
            tree=tree,
            scene_points=scene_points,
            intrinsics=intrinsics,
            frame_id=int(frame_id),
            xy=src_xy[np.asarray(carrier_indices, dtype=np.int64)],
            nn_radius=float(nn_radius),
        )
        for local_idx, point_id in zip(input_indices.tolist(), hits.tolist()):
            if carrier_indices[int(local_idx)] in component_set:
                point_ids.add(int(point_id))
    return point_ids


def _backproject_mask_points(exporter, selected_masks: list[tuple[int, int, int, float]], nn_radius: float) -> tuple[set[int], int]:
    point_ids: set[int] = set()
    queries = 0
    for frame_id, mask_id, _, _ in selected_masks:
        hits, query_count = exporter._backproject_mask(int(frame_id), int(mask_id), nn_radius=float(nn_radius))
        queries += int(query_count)
        point_ids.update(int(v) for v in hits.tolist())
    return point_ids, queries


def _uses_full_mask_fringe(support_mode: str) -> bool:
    return support_mode in {"core_fringe", "core_fringe_wta"}


def _uses_owned_fringe(support_mode: str) -> bool:
    return support_mode in {
        "core_owned_fringe",
        "core_owned_fringe_wta",
        "core_owned_track_fringe",
        "core_owned_track_fringe_wta",
        "core_owned_fringe_wta_posttrack",
    }


def _uses_seeded_fringe(support_mode: str) -> bool:
    return support_mode in {"seeded_fringe", "seeded_fringe_wta"}


def _uses_wta(support_mode: str) -> bool:
    return support_mode in {
        "core_fringe_wta",
        "core_owned_fringe_wta",
        "core_owned_track_fringe_wta",
        "core_owned_fringe_wta_posttrack",
        "seeded_fringe_wta",
    }


def _uses_mask_support(support_mode: str) -> bool:
    return _uses_full_mask_fringe(support_mode) or _uses_owned_fringe(support_mode) or _uses_seeded_fringe(support_mode)


def _uses_scene_track_linking(support_mode: str) -> bool:
    return support_mode in {"core_owned_track_fringe", "core_owned_track_fringe_wta"}


def _uses_post_wta_scene_track_linking(support_mode: str) -> bool:
    return support_mode in {"core_owned_fringe_wta_posttrack"}


def _build_mask_ownership(component_infos: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], int], dict[str, float]]:
    best: dict[tuple[int, int], tuple[tuple[float, float, float, float, int], int]] = {}
    claim_counts: Counter[tuple[int, int]] = Counter()
    total_claims = 0
    for component_idx, info in enumerate(component_infos):
        component = info["component"]
        frames = info["frames"]
        component_size = float(max(len(component), 1))
        frame_count = float(max(len(frames), 1))
        for frame_id, mask_id, count, ratio in info["selected_masks"]:
            key = (int(frame_id), int(mask_id))
            total_claims += 1
            claim_counts[key] += 1
            score = float(count) * float(ratio) * np.sqrt(component_size) * np.sqrt(frame_count)
            candidate = (score, float(count), float(ratio), component_size, -int(component_idx))
            if key not in best or candidate > best[key][0]:
                best[key] = (candidate, int(component_idx))
    ownership = {key: int(component_idx) for key, (_, component_idx) in best.items()}
    competing = sum(1 for value in claim_counts.values() if int(value) > 1)
    return ownership, {
        "ownership_candidate_mask_claims": float(total_claims),
        "ownership_unique_masks": float(len(ownership)),
        "ownership_competing_masks": float(competing),
    }


def _record_mask_frame_sets(object_info: dict) -> dict[int, set[int]]:
    frame_masks: dict[int, set[int]] = {}
    for item in object_info.get("mask_list", []):
        if len(item) < 2:
            continue
        frame_id, mask_id = int(item[0]), int(item[1])
        frame_masks.setdefault(frame_id, set()).add(mask_id)
    return frame_masks


def _merge_scene_track_records(
    records: list[dict],
    object_dict: dict[int, dict],
    args: argparse.Namespace,
) -> tuple[list[dict], dict[int, dict], dict[str, float]]:
    if not records:
        return records, object_dict, {
            "scene_link_candidate_pairs": 0.0,
            "scene_link_accepted_pairs": 0.0,
            "scene_link_output_records": 0.0,
        }

    record_ids = [int(record["object_id"]) for record in records]
    record_pos = {record_id: pos for pos, record_id in enumerate(record_ids)}
    carrier_sets: dict[int, set[int]] = {}
    frame_masks: dict[int, dict[int, set[int]]] = {}
    window_indices: dict[int, int] = {}
    carrier_to_records: dict[int, list[int]] = {}
    for record_id in record_ids:
        obj = object_dict[int(record_id)]
        carriers = set(int(v) for v in np.asarray(obj.get("carrier_ids", []), dtype=np.int64).reshape(-1).tolist())
        carrier_sets[int(record_id)] = carriers
        frame_masks[int(record_id)] = _record_mask_frame_sets(obj)
        window_indices[int(record_id)] = int(obj.get("window_index", -1))
        for carrier_id in carriers:
            carrier_to_records.setdefault(int(carrier_id), []).append(int(record_id))

    pair_shared: Counter[tuple[int, int]] = Counter()
    for ids in carrier_to_records.values():
        ids = sorted(set(int(v) for v in ids))
        for pos, left in enumerate(ids):
            for right in ids[pos + 1 :]:
                left_window = window_indices.get(int(left), -1)
                right_window = window_indices.get(int(right), -1)
                if left_window == right_window:
                    continue
                max_gap = int(args.scene_link_max_window_gap)
                if max_gap > 0 and abs(left_window - right_window) > max_gap:
                    continue
                pair_shared[(int(left), int(right))] += 1

    min_shared = int(args.scene_link_min_shared_carriers)
    min_ratio = float(args.scene_link_min_overlap_ratio)
    candidates: list[tuple[float, int, int, int]] = []
    for (left, right), shared in pair_shared.items():
        denom = max(1, min(len(carrier_sets[int(left)]), len(carrier_sets[int(right)])))
        ratio = float(shared / denom)
        if int(shared) >= min_shared and ratio >= min_ratio:
            candidates.append((ratio, int(shared), int(left), int(right)))
    candidates.sort(reverse=True)

    parent = {record_id: record_id for record_id in record_ids}
    group_records = {record_id: {record_id} for record_id in record_ids}
    group_frame_masks = {record_id: {frame: set(masks) for frame, masks in frame_masks[record_id].items()} for record_id in record_ids}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def can_union(left_root: int, right_root: int) -> bool:
        max_masks = int(args.scene_link_max_masks_per_frame)
        if max_masks <= 0:
            return True
        merged_frames = set(group_frame_masks[left_root]).union(group_frame_masks[right_root])
        for frame_id in merged_frames:
            masks = group_frame_masks[left_root].get(frame_id, set()) | group_frame_masks[right_root].get(frame_id, set())
            if len(masks) > max_masks:
                return False
        return True

    accepted = 0
    rejected_frame_conflict = 0
    rejected_same_group = 0
    for _, _, left, right in candidates:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            rejected_same_group += 1
            continue
        if not can_union(left_root, right_root):
            rejected_frame_conflict += 1
            continue
        if len(group_records[left_root]) < len(group_records[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        group_records[left_root].update(group_records.pop(right_root))
        for frame_id, masks in group_frame_masks.pop(right_root).items():
            group_frame_masks[left_root].setdefault(frame_id, set()).update(masks)
        accepted += 1

    groups = [sorted(items) for root, items in sorted(group_records.items()) if find(root) == root]
    record_by_id = {int(record["object_id"]): record for record in records}
    merged_records: list[dict] = []
    merged_objects: dict[int, dict] = {}
    for new_id, members in enumerate(groups):
        point_ids: set[int] = set()
        carrier_ids: set[int] = set()
        source_windows: set[int] = set()
        mask_best: dict[tuple[int, int], float] = {}
        for old_id in members:
            old_record = record_by_id[int(old_id)]
            old_obj = object_dict[int(old_id)]
            point_ids.update(int(v) for v in old_record["point_ids"])
            carrier_ids.update(int(v) for v in carrier_sets[int(old_id)])
            source_windows.add(int(window_indices.get(int(old_id), -1)))
            for item in old_obj.get("mask_list", []):
                if len(item) < 3:
                    continue
                key = (int(item[0]), int(item[1]))
                mask_best[key] = max(float(item[2]), mask_best.get(key, 0.0))
        mask_list = [(frame_id, mask_id, ratio) for (frame_id, mask_id), ratio in sorted(mask_best.items())]
        frames = {int(item[0]) for item in mask_list}
        area = float(len(point_ids))
        observations = float(len(mask_list))
        reliability = observations * np.sqrt(max(area, 1.0)) * np.sqrt(max(len(frames), 1))
        record = {
            "object_id": int(new_id),
            "point_ids": point_ids,
            "area_score": area,
            "score": area,
            "observations": observations,
            "carrier_count": float(len(carrier_ids)),
            "frame_count": float(len(frames)),
            "reliability": float(reliability),
            "dense_quality": float(reliability),
            "selection_quality": float(reliability),
            "source_record_count": float(len(members)),
        }
        merged_records.append(record)
        merged_objects[int(new_id)] = {
            "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
            "mask_list": mask_list,
            "repre_mask_list": sorted(mask_list, key=lambda item: item[2], reverse=True)[:5],
            "carrier_ids": np.asarray(sorted(carrier_ids), dtype=np.int64),
            "source_object_ids": np.asarray(sorted(members), dtype=np.int64),
            "source_window_indices": np.asarray(sorted(source_windows), dtype=np.int64),
        }

    group_sizes = [len(group) for group in groups]
    merged_groups = sum(1 for size in group_sizes if size > 1)
    return merged_records, merged_objects, {
        "scene_link_candidate_pairs_raw": float(len(pair_shared)),
        "scene_link_candidate_pairs": float(len(candidates)),
        "scene_link_accepted_pairs": float(accepted),
        "scene_link_rejected_same_group": float(rejected_same_group),
        "scene_link_rejected_frame_mask_conflict": float(rejected_frame_conflict),
        "scene_link_input_records": float(len(records)),
        "scene_link_output_records": float(len(merged_records)),
        "scene_link_merged_groups": float(merged_groups),
        "scene_link_max_group_size": float(max(group_sizes) if group_sizes else 0),
        "scene_link_mean_group_size": float(np.mean(group_sizes) if group_sizes else 0.0),
    }


def _build_seeded_densifier(stream: ScanNetStream, exporter, args: argparse.Namespace) -> ReliableDensifier:
    params = ReliableDensifyParams(
        max_masks_per_object=int(args.seeded_max_masks_per_object),
        mask_min_relative_coverage=float(args.seeded_mask_min_relative_coverage),
        mask_sample_stride=int(args.seeded_mask_sample_stride),
        mask_max_pixels=int(args.seeded_mask_max_pixels),
        boundary_erosion=int(args.seeded_boundary_erosion),
        small_mask_area=int(args.seeded_small_mask_area),
        seed_distance_px=float(args.seeded_distance_px),
        min_seed_pixels=int(args.seeded_min_seed_pixels),
        nn_radius=float(args.mask_nn_radius),
        seed_keep_mode=str(args.seeded_seed_keep_mode),
        seed_min_support_views=int(args.seeded_seed_min_support_views),
        mask_selection_mode=str(args.seeded_mask_selection_mode),
    )
    return ReliableDensifier(
        stream=stream,
        scene_points=exporter.scene_points,
        tree=exporter.tree,
        intrinsics=exporter.intrinsics,
        params=params,
    )


def _seeded_mask_points(
    densifier: ReliableDensifier,
    core_points: set[int],
    selected_masks: list[tuple[int, int, int, float]],
) -> tuple[set[int], dict[str, float]]:
    mask_list = [(frame_id, mask_id, ratio) for frame_id, mask_id, _, ratio in selected_masks]
    result = densifier.densify_object(
        {
            "point_ids": np.asarray(sorted(core_points), dtype=np.int64),
            "mask_list": mask_list,
        }
    )
    return set(int(v) for v in result.point_ids), {str(k): float(v) for k, v in result.diagnostics.items()}


def _component_records(
    stream: ScanNetStream,
    exporter,
    window: dict[str, Any],
    components: list[list[int]],
    args: argparse.Namespace,
    seeded_densifier: ReliableDensifier | None = None,
) -> tuple[list[dict], dict[int, dict], dict[str, float]]:
    records: list[dict] = []
    object_dict: dict[int, dict] = {}
    totals = Counter()
    seeded_diag_totals = Counter()
    component_infos: list[dict[str, Any]] = []
    for component in components:
        frames = {frame_id for idx in component for frame_id in window["memberships"][idx]}
        if len(component) < int(args.min_component_carriers) or len(frames) < int(args.min_component_frames):
            totals["dropped_small_component"] += 1
            continue
        selected_masks = _selected_masks_for_component(
            window["memberships"],
            component,
            min_mask_carriers=int(args.min_mask_carriers),
            min_frame_mask_ratio=float(args.min_frame_mask_ratio),
        )
        component_infos.append(
            {
                "component": component,
                "frames": frames,
                "selected_masks": selected_masks,
            }
        )

    ownership: dict[tuple[int, int], int] = {}
    if _uses_owned_fringe(args.support_mode):
        ownership, ownership_diag = _build_mask_ownership(component_infos)
        totals.update(ownership_diag)

    for component_idx, info in enumerate(component_infos):
        component = info["component"]
        frames = info["frames"]
        selected_masks = list(info["selected_masks"])
        if _uses_owned_fringe(args.support_mode):
            before = len(selected_masks)
            selected_masks = [
                item
                for item in selected_masks
                if ownership.get((int(item[0]), int(item[1]))) == int(component_idx)
            ]
            totals["ownership_mask_claims_kept"] += len(selected_masks)
            totals["ownership_mask_claims_dropped"] += before - len(selected_masks)
        core_points = _core_points_for_component(
            stream,
            exporter.tree,
            exporter.scene_points,
            exporter.intrinsics,
            window,
            component,
            nn_radius=float(args.core_nn_radius),
        )
        mask_points: set[int] = set()
        mask_queries = 0
        if _uses_full_mask_fringe(args.support_mode) or _uses_owned_fringe(args.support_mode):
            mask_points, mask_queries = _backproject_mask_points(exporter, selected_masks, nn_radius=float(args.mask_nn_radius))
        elif _uses_seeded_fringe(args.support_mode):
            if seeded_densifier is None:
                raise RuntimeError("seeded_densifier is required for seeded fringe support modes")
            seeded_points, seeded_diag = _seeded_mask_points(seeded_densifier, core_points, selected_masks)
            mask_points = seeded_points
            mask_queries = int(seeded_diag.get("densify_backproject_queries", 0.0))
            for key, value in seeded_diag.items():
                seeded_diag_totals[f"seeded_{key}"] += float(value)
        point_ids = set(core_points)
        if _uses_mask_support(args.support_mode):
            point_ids.update(mask_points)
        if len(point_ids) < int(args.min_points_per_object):
            totals["dropped_small_object"] += 1
            continue
        object_id = len(records)
        observations = float(len(selected_masks))
        area = float(len(point_ids))
        reliability = observations * np.sqrt(max(area, 1.0)) * np.sqrt(max(len(frames), 1))
        mask_list = [(frame_id, mask_id, ratio) for frame_id, mask_id, _, ratio in selected_masks]
        record = {
            "object_id": int(object_id),
            "point_ids": point_ids,
            "area_score": area,
            "score": area,
            "observations": observations,
            "carrier_count": float(len(component)),
            "frame_count": float(len(frames)),
            "reliability": float(reliability),
            "dense_quality": float(reliability),
            "selection_quality": float(reliability),
        }
        records.append(record)
        object_dict[int(object_id)] = {
            "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
            "mask_list": sorted(mask_list, key=lambda item: (item[0], item[1])),
            "repre_mask_list": sorted(mask_list, key=lambda item: item[2], reverse=True)[:5],
            "carrier_ids": np.asarray(window["carrier_id"][np.asarray(component, dtype=np.int64)], dtype=np.int64),
        }
        totals["core_points"] += len(core_points)
        totals["fringe_candidate_points"] += len(mask_points)
        totals["mask_backproject_queries"] += mask_queries
        totals["selected_masks"] += len(selected_masks)
        totals["component_carriers"] += len(component)
    diag = {f"support_{key}": float(value) for key, value in totals.items()}
    diag.update({f"support_{key}": float(value) for key, value in seeded_diag_totals.items()})
    return records, object_dict, diag


def _write_prediction(exporter, records: list[dict], object_dict: dict[int, dict], args: argparse.Namespace) -> dict[str, float]:
    wta_diag: dict[str, float] = {}
    if _uses_wta(args.support_mode):
        records, wta_diag = apply_wta_to_records(records)
        for record in records:
            object_id = int(record["object_id"])
            if object_id in object_dict:
                object_dict[object_id]["point_ids"] = np.asarray(sorted(record["point_ids"]), dtype=np.int64)
    post_link_diag: dict[str, float] = {}
    if _uses_post_wta_scene_track_linking(args.support_mode):
        records, object_dict, post_link_diag = _merge_scene_track_records(records, object_dict, args)

    kept_records = [record for record in records if len(record["point_ids"]) >= int(args.min_points_per_object)]
    masks = np.zeros((exporter.scene_points.shape[0], len(kept_records)), dtype=bool)
    scores = np.zeros((len(kept_records),), dtype=np.float32)
    out_object_dict: dict[int, dict] = {}
    for out_idx, record in enumerate(kept_records):
        point_ids = np.asarray(sorted(record["point_ids"]), dtype=np.int64)
        masks[point_ids, out_idx] = True
        scores[out_idx] = float(record.get("reliability", len(point_ids)))
        old_object = object_dict[int(record["object_id"])]
        out_object_dict[out_idx] = {
            **old_object,
            "point_ids": point_ids,
        }
    pred_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{args.seq_name}.npz",
        pred_masks=masks,
        pred_score=scores,
        pred_classes=np.zeros((scores.shape[0],), dtype=np.int32),
    )
    tmp_dir = Path("data/TMP") / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_points = np.flatnonzero(masks.any(axis=1)).astype(np.int64)
    np.save(tmp_dir / f"{args.seq_name}_pre_points.npy", pre_points)
    object_dir = exporter.stream.object_dir / args.output_config
    object_dict_path = object_dir / "object_dict.npy"
    object_dict_write_fallback = False
    object_dict_write_error = ""
    try:
        object_dir.mkdir(parents=True, exist_ok=True)
        np.save(object_dict_path, out_object_dict, allow_pickle=True)
    except OSError as exc:
        object_dict_write_fallback = True
        object_dict_write_error = f"{type(exc).__name__}: {exc}"
        fallback_dir = Path(args.summary_root) / "object_dicts" / args.output_config / args.seq_name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        object_dict_path = fallback_dir / "object_dict.npy"
        np.save(object_dict_path, out_object_dict, allow_pickle=True)
    owner_counts = masks.sum(axis=1) if masks.size else np.zeros((exporter.scene_points.shape[0],), dtype=np.int16)
    return {
        "num_exported_objects": float(len(kept_records)),
        "num_exported_points": float(pre_points.shape[0]),
        "export_conflict_rate": float(np.count_nonzero(owner_counts > 1) / max(pre_points.shape[0], 1)),
        "object_dict_write_fallback": bool(object_dict_write_fallback),
        "object_dict_write_path": str(object_dict_path),
        "object_dict_write_error": object_dict_write_error,
        **wta_diag,
        **post_link_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v7 carrier-tracklet graph predictions from cached D4RT carriers.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--support-mode",
        choices=[
            "core",
            "core_fringe",
            "core_fringe_wta",
            "core_owned_fringe",
            "core_owned_fringe_wta",
            "core_owned_track_fringe",
            "core_owned_track_fringe_wta",
            "core_owned_fringe_wta_posttrack",
            "seeded_fringe",
            "seeded_fringe_wta",
        ],
        default="core",
    )
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-visibility", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-shared-frames", type=int, default=2)
    parser.add_argument("--min-positive-ratio", type=float, default=0.45)
    parser.add_argument("--max-carriers-per-mask-edge", type=int, default=160)
    parser.add_argument("--max-pair-distance-variance", type=float, default=0.02)
    parser.add_argument("--max-masks-per-frame-component", type=int, default=1)
    parser.add_argument("--max-component-carriers", type=int, default=500)
    parser.add_argument("--min-component-carriers", type=int, default=8)
    parser.add_argument("--min-component-frames", type=int, default=2)
    parser.add_argument("--min-mask-carriers", type=int, default=4)
    parser.add_argument("--min-frame-mask-ratio", type=float, default=0.50)
    parser.add_argument("--min-points-per-object", type=int, default=80)
    parser.add_argument("--core-nn-radius", type=float, default=0.05)
    parser.add_argument("--mask-nn-radius", type=float, default=0.05)
    parser.add_argument("--scene-link-min-shared-carriers", type=int, default=10)
    parser.add_argument("--scene-link-min-overlap-ratio", type=float, default=0.15)
    parser.add_argument("--scene-link-max-window-gap", type=int, default=1)
    parser.add_argument("--scene-link-max-masks-per-frame", type=int, default=1)
    parser.add_argument("--seeded-max-masks-per-object", type=int, default=8)
    parser.add_argument("--seeded-mask-min-relative-coverage", type=float, default=0.0)
    parser.add_argument("--seeded-mask-sample-stride", type=int, default=1)
    parser.add_argument("--seeded-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--seeded-boundary-erosion", type=int, default=1)
    parser.add_argument("--seeded-small-mask-area", type=int, default=400)
    parser.add_argument("--seeded-distance-px", type=float, default=32.0)
    parser.add_argument("--seeded-min-seed-pixels", type=int, default=1)
    parser.add_argument("--seeded-seed-keep-mode", choices=["none", "supported", "boundary", "component", "all"], default="all")
    parser.add_argument("--seeded-seed-min-support-views", type=int, default=1)
    parser.add_argument(
        "--seeded-mask-selection-mode",
        choices=[
            "coverage",
            "seed_density",
            "component_seed_density",
            "kept_seed_density",
            "coverage_component_density",
            "coverage_kept_density",
            "kept_ratio",
        ],
        default="coverage_kept_density",
    )
    parser.add_argument("--summary-root", default="outputs/v7_carrier_tracklet_graph")
    args = parser.parse_args()

    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))

    from stream4d.export_scannet import ScanNetExporter

    exporter = ScanNetExporter(stream, output_config=args.output_config, export_nn_radius=float(args.mask_nn_radius))
    seeded_densifier = _build_seeded_densifier(stream, exporter, args) if _uses_seeded_fringe(args.support_mode) else None
    scene_dir = Path(args.debug_root) / args.seq_name
    carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
    if not carrier_paths:
        raise FileNotFoundError(f"no carriers_window*.npz under {scene_dir}")

    all_records: list[dict] = []
    all_objects: dict[int, dict] = {}
    scene_diag = Counter()
    window_summaries: list[dict[str, Any]] = []
    for window_idx, carrier_path in enumerate(carrier_paths):
        window = _window_memberships(
            stream,
            carrier_path,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
        )
        components, cluster_diag = _cluster_window(window, args)
        records, object_dict, support_diag = _component_records(
            stream,
            exporter,
            window,
            components,
            args,
            seeded_densifier=seeded_densifier,
        )
        offset = len(all_records)
        for record in records:
            new_record = dict(record)
            new_id = offset + int(record["object_id"])
            new_record["object_id"] = new_id
            new_record["window_index"] = int(window_idx)
            all_records.append(new_record)
            all_objects[new_id] = {
                **object_dict[int(record["object_id"])],
                "window_index": int(window_idx),
            }
        for key, value in {**cluster_diag, **support_diag}.items():
            scene_diag[key] += float(value)
        window_summaries.append(
            {
                "carrier_file": str(carrier_path),
                "num_carriers": int(window["num_carriers"]),
                "valid_assignments": int(window["valid_assignments"]),
                "components": int(len(components)),
                "records": int(len(records)),
                **cluster_diag,
                **support_diag,
            }
        )

    scene_link_diag: dict[str, float] = {}
    if _uses_scene_track_linking(args.support_mode):
        all_records, all_objects, scene_link_diag = _merge_scene_track_records(all_records, all_objects, args)
        for key, value in scene_link_diag.items():
            scene_diag[key] += float(value)

    export_diag = _write_prediction(exporter, all_records, all_objects, args)
    summary = {
        "args": vars(args),
        "algorithm": "carrier_tracklet_comembership_graph_v7",
        "uses_mask_node_point_overlap": False,
        "uses_gt": False,
        "num_windows": len(carrier_paths),
        "num_candidate_records": int(len(all_records)),
        **{key: float(value) for key, value in scene_diag.items()},
        **export_diag,
        "windows": window_summaries,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.output_config}_{args.seq_name}_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.debug_root)],
        pre_points_policy="recompute",
        support_policy=f"carrier_tracklet_graph_v7:{args.support_mode}",
        notes=(
            "Carrier-tracklet graph v7: carrier co-membership edges from cached D4RT carrier trajectories; "
            "owned modes assign each frame mask to one carrier component before support expansion; "
            "track modes merge cross-window records by shared carrier IDs before WTA; "
            "seeded modes densify only connected RGB-D mask regions around carrier core seeds; no GT used."
        ),
        extra={
            "algorithm": summary["algorithm"],
            "uses_mask_node_point_overlap": False,
            "summary_path": str(summary_path),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
