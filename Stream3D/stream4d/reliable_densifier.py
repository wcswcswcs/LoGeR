from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ReliableDensifyParams:
    max_masks_per_object: int = 5
    mask_min_relative_coverage: float = 0.0
    mask_sample_stride: int = 1
    mask_max_pixels: int = 50000
    boundary_erosion: int = 1
    small_mask_area: int = 400
    seed_distance_px: float = 32.0
    min_seed_pixels: int = 1
    nn_radius: float = 0.08
    seed_keep_mode: str = "none"
    seed_min_support_views: int = 1
    mask_selection_mode: str = "coverage"


@dataclass
class ReliableDensifyResult:
    point_ids: set[int]
    diagnostics: dict[str, float]


def unique_mask_observations(mask_observations: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    best: dict[tuple[int, int], float] = {}
    for frame_id, mask_id, coverage in mask_observations:
        key = (int(frame_id), int(mask_id))
        best[key] = max(float(coverage), best.get(key, 0.0))
    out = [(frame_id, mask_id, coverage) for (frame_id, mask_id), coverage in best.items()]
    return sorted(out, key=lambda item: float(item[2]), reverse=True)


def recompute_record_scores(record: dict, old_area: float | None = None) -> dict:
    """Refresh area-sensitive scores after a record's point assignment changes."""
    new_record = dict(record)
    point_ids = new_record.get("point_ids", [])
    area = float(len(point_ids))
    old_area_value = float(old_area if old_area is not None else new_record.get("area_score", area))
    old_area_value = max(old_area_value, 1.0)
    area_scale = float(np.sqrt(max(area, 1.0) / old_area_value))

    observations = float(new_record.get("observations", 0.0))
    old_reliability = float(new_record.get("reliability", area))
    old_dense_quality = float(new_record.get("dense_quality", old_reliability))
    old_selection_quality = float(new_record.get("selection_quality", old_dense_quality))

    new_record["area_score"] = area
    new_record["score"] = area
    if observations > 0.0:
        new_record["reliability"] = float(observations * np.sqrt(max(area, 1.0)))
    else:
        new_record["reliability"] = float(old_reliability * area_scale)
    new_record["dense_quality"] = float(old_dense_quality * area_scale)
    new_record["selection_quality"] = float(old_selection_quality * area_scale)
    return new_record


def apply_wta_to_records(
    object_records: list[dict],
    min_conflict_owners: int = 2,
) -> tuple[list[dict], dict[str, float]]:
    min_conflict_owners = max(2, int(min_conflict_owners))
    total_assignments = int(sum(len(record["point_ids"]) for record in object_records))
    counts: Counter[int] = Counter()
    owner: dict[int, tuple[float, int]] = {}
    for record_idx, record in enumerate(object_records):
        reliability = float(record.get("reliability", len(record["point_ids"])))
        for point_id in record["point_ids"]:
            pid = int(point_id)
            counts[pid] += 1
            prev = owner.get(pid)
            if prev is None or reliability > prev[0] or (reliability == prev[0] and record_idx < prev[1]):
                owner[pid] = (reliability, record_idx)

    reassigned = [set() for _ in object_records]
    resolved_points = 0
    for record_idx, record in enumerate(object_records):
        for point_id in record["point_ids"]:
            pid = int(point_id)
            if counts[pid] >= min_conflict_owners:
                continue
            reassigned[record_idx].add(pid)
    for point_id, (_, record_idx) in owner.items():
        if counts[int(point_id)] < min_conflict_owners:
            continue
        reassigned[record_idx].add(int(point_id))
        resolved_points += 1

    out: list[dict] = []
    for record, point_ids in zip(object_records, reassigned):
        new_record = dict(record)
        old_area = float(len(record["point_ids"]))
        new_record["point_ids"] = point_ids
        out.append(recompute_record_scores(new_record, old_area=old_area))

    conflict_points = int(sum(1 for value in counts.values() if value > 1))
    unique_points = int(len(counts))
    removed_assignments = int(total_assignments - sum(len(record["point_ids"]) for record in out))
    return out, {
        "densify_wta_total_assignments": float(total_assignments),
        "densify_wta_unique_points": float(unique_points),
        "densify_wta_conflict_points": float(conflict_points),
        "densify_wta_resolved_points": float(resolved_points),
        "densify_wta_min_conflict_owners": float(min_conflict_owners),
        "densify_wta_removed_assignments": float(removed_assignments),
        "densify_wta_pre_conflict_rate": float(conflict_points / max(unique_points, 1)),
        "densify_wta_removed_assignment_rate": float(removed_assignments / max(total_assignments, 1)),
    }


class ReliableDensifier:
    def __init__(
        self,
        stream,
        scene_points: np.ndarray,
        tree: cKDTree,
        intrinsics: np.ndarray,
        params: ReliableDensifyParams,
    ) -> None:
        self.stream = stream
        self.scene_points = np.asarray(scene_points, dtype=np.float32)
        self.tree = tree
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.params = params
        self._depth_cache: dict[int, np.ndarray] = {}
        self._mask_cache: dict[int, np.ndarray] = {}
        self._pose_inv_cache: dict[int, np.ndarray | None] = {}

    def _depth(self, frame_id: int) -> np.ndarray:
        frame_id = int(frame_id)
        if frame_id not in self._depth_cache:
            self._depth_cache[frame_id] = self.stream.load_depth(frame_id)
        return self._depth_cache[frame_id]

    def _mask(self, frame_id: int) -> np.ndarray:
        frame_id = int(frame_id)
        if frame_id not in self._mask_cache:
            depth = self._depth(frame_id)
            mask = self.stream.load_mask(frame_id)
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._mask_cache[frame_id] = mask
        return self._mask_cache[frame_id]

    def _pose_inv(self, frame_id: int) -> np.ndarray | None:
        frame_id = int(frame_id)
        if frame_id not in self._pose_inv_cache:
            pose = self.stream.load_pose(frame_id)
            if not np.isfinite(pose).all():
                self._pose_inv_cache[frame_id] = None
            else:
                try:
                    self._pose_inv_cache[frame_id] = np.linalg.inv(pose).astype(np.float32)
                except np.linalg.LinAlgError:
                    self._pose_inv_cache[frame_id] = None
        return self._pose_inv_cache[frame_id]

    def _project_points(self, frame_id: int, point_ids: np.ndarray) -> np.ndarray:
        projected, _ = self._project_points_with_ids(frame_id, point_ids)
        return projected

    def _project_points_with_ids(self, frame_id: int, point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if point_ids.size == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
        depth = self._depth(frame_id)
        h, w = depth.shape
        pose_inv = self._pose_inv(frame_id)
        if pose_inv is None:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)

        point_ids = point_ids[(point_ids >= 0) & (point_ids < self.scene_points.shape[0])]
        if point_ids.size == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
        pts = self.scene_points[point_ids]
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        cam = (pose_inv @ pts_h.T).T[:, :3]
        z = cam[:, 2]
        valid = np.isfinite(cam).all(axis=1) & (z > 1e-6)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
        valid_point_ids = point_ids[valid]
        cam = cam[valid]
        z = cam[:, 2]
        fx = float(self.intrinsics[0, 0])
        fy = float(self.intrinsics[1, 1])
        cx = float(self.intrinsics[0, 2])
        cy = float(self.intrinsics[1, 2])
        xs = np.rint((cam[:, 0] * fx / z) + cx).astype(np.int64)
        ys = np.rint((cam[:, 1] * fy / z) + cy).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(in_bounds):
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
        xy = np.stack([xs[in_bounds], ys[in_bounds]], axis=1).astype(np.int64)
        return xy, valid_point_ids[in_bounds].astype(np.int64)

    @staticmethod
    def _component_from_seeds(mask_bool: np.ndarray, seed_xy: np.ndarray) -> tuple[np.ndarray, int]:
        if seed_xy.size == 0 or not np.any(mask_bool):
            return np.zeros_like(mask_bool, dtype=bool), 0
        labels_count, labels = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
        if labels_count <= 1:
            return np.zeros_like(mask_bool, dtype=bool), 0
        seed_labels = labels[seed_xy[:, 1], seed_xy[:, 0]]
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
            return np.zeros_like(mask_bool, dtype=bool), 0
        keep_labels = np.unique(seed_labels)
        return np.isin(labels, keep_labels), int(keep_labels.shape[0])

    @staticmethod
    def _distance_filter(component: np.ndarray, seed_xy: np.ndarray, max_distance: float) -> np.ndarray:
        if max_distance <= 0.0 or seed_xy.size == 0 or not np.any(component):
            return component
        distance_input = np.ones(component.shape, dtype=np.uint8)
        distance_input[seed_xy[:, 1], seed_xy[:, 0]] = 0
        distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
        return component & (distance <= float(max_distance))

    def _score_observation_selection(
        self,
        frame_id: int,
        mask_id: int,
        coverage: float,
        seed_point_ids: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        if self.params.mask_selection_mode == "coverage":
            return float(coverage), {
                "selection_seed_pixels": 0.0,
                "selection_component_pixels": 0.0,
                "selection_distance_pixels": 0.0,
            }
        mask = self._mask(int(frame_id))
        mask_bool = mask == int(mask_id)
        mask_area = int(np.count_nonzero(mask_bool))
        if mask_area <= 0:
            return -1.0, {
                "selection_seed_pixels": 0.0,
                "selection_component_pixels": 0.0,
                "selection_distance_pixels": 0.0,
            }
        projected = self._project_points(int(frame_id), seed_point_ids)
        if projected.size == 0:
            return -1.0, {
                "selection_seed_pixels": 0.0,
                "selection_component_pixels": 0.0,
                "selection_distance_pixels": 0.0,
            }
        inside = mask_bool[projected[:, 1], projected[:, 0]]
        seed_xy = np.unique(projected[inside], axis=0)
        seed_pixels = int(seed_xy.shape[0])
        if seed_pixels < int(self.params.min_seed_pixels):
            return -1.0, {
                "selection_seed_pixels": float(seed_pixels),
                "selection_component_pixels": 0.0,
                "selection_distance_pixels": 0.0,
            }
        component, _ = self._component_from_seeds(mask_bool, seed_xy)
        component_pixels = int(np.count_nonzero(component))
        if component_pixels <= 0:
            return -1.0, {
                "selection_seed_pixels": float(seed_pixels),
                "selection_component_pixels": 0.0,
                "selection_distance_pixels": 0.0,
            }
        filtered = component
        erosion = int(self.params.boundary_erosion)
        if erosion > 0 and component_pixels >= int(self.params.small_mask_area):
            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = cv2.erode(component.astype(np.uint8), kernel, iterations=erosion).astype(bool)
            if np.any(eroded) and np.any(eroded[seed_xy[:, 1], seed_xy[:, 0]]):
                filtered = eroded
        distance_filtered = self._distance_filter(filtered, seed_xy, float(self.params.seed_distance_px))
        distance_pixels = int(np.count_nonzero(distance_filtered))
        seed_density = float(seed_pixels / max(mask_area, 1))
        component_density = float(seed_pixels / max(component_pixels, 1))
        kept_density = float(seed_pixels / max(distance_pixels, 1))
        kept_ratio = float(distance_pixels / max(mask_area, 1))
        if self.params.mask_selection_mode == "seed_density":
            score = seed_density
        elif self.params.mask_selection_mode == "component_seed_density":
            score = component_density
        elif self.params.mask_selection_mode == "kept_seed_density":
            score = kept_density
        elif self.params.mask_selection_mode == "coverage_component_density":
            score = float(coverage) * component_density
        elif self.params.mask_selection_mode == "coverage_kept_density":
            score = float(coverage) * kept_density
        elif self.params.mask_selection_mode == "kept_ratio":
            score = kept_ratio
        else:
            raise ValueError(f"Unsupported mask_selection_mode: {self.params.mask_selection_mode}")
        return float(score), {
            "selection_seed_pixels": float(seed_pixels),
            "selection_component_pixels": float(component_pixels),
            "selection_distance_pixels": float(distance_pixels),
        }

    def _select_observations(
        self,
        observations: list[tuple[int, int, float]],
        seed_point_ids: np.ndarray,
        diag: dict[str, float],
    ) -> list[tuple[int, int, float]]:
        if not observations:
            return observations
        if self.params.mask_selection_mode != "coverage":
            scored: list[tuple[float, tuple[int, int, float], dict[str, float]]] = []
            for frame_id, mask_id, coverage in observations:
                score, selection_diag = self._score_observation_selection(
                    int(frame_id),
                    int(mask_id),
                    float(coverage),
                    seed_point_ids,
                )
                scored.append((score, (int(frame_id), int(mask_id), float(coverage)), selection_diag))
            scored.sort(key=lambda item: (item[0], item[1][2]), reverse=True)
            observations = [item[1] for item in scored if item[0] >= 0.0]
            diag["densify_selection_score_mean"] = float(np.mean([item[0] for item in scored])) if scored else 0.0
            diag["densify_selection_score_max"] = float(np.max([item[0] for item in scored])) if scored else 0.0
            diag["densify_selection_seed_pixels_sum"] = float(
                sum(item[2].get("selection_seed_pixels", 0.0) for item in scored)
            )
            diag["densify_selection_component_pixels_sum"] = float(
                sum(item[2].get("selection_component_pixels", 0.0) for item in scored)
            )
            diag["densify_selection_distance_pixels_sum"] = float(
                sum(item[2].get("selection_distance_pixels", 0.0) for item in scored)
            )
            score_by_key = {(int(item[1][0]), int(item[1][1])): float(item[0]) for item in scored}
        else:
            score_by_key = {(int(frame_id), int(mask_id)): float(coverage) for frame_id, mask_id, coverage in observations}

        min_relative_coverage = max(0.0, float(self.params.mask_min_relative_coverage))
        if observations and min_relative_coverage > 0.0:
            top_value = max(float(observations[0][2]), 1e-12)
            if self.params.mask_selection_mode != "coverage":
                top_value = max(
                    self._score_observation_selection(
                        int(observations[0][0]),
                        int(observations[0][1]),
                        float(observations[0][2]),
                        seed_point_ids,
                    )[0],
                    1e-12,
                )
                observations = [
                    item
                    for item in observations
                    if self._score_observation_selection(int(item[0]), int(item[1]), float(item[2]), seed_point_ids)[0]
                    >= top_value * min_relative_coverage
                ]
            else:
                observations = [
                    item for item in observations if float(item[2]) >= top_value * min_relative_coverage
                ]
        if self.params.max_masks_per_object > 0:
            observations = observations[: int(self.params.max_masks_per_object)]
        selected_scores = [score_by_key.get((int(item[0]), int(item[1])), 0.0) for item in observations]
        diag["densify_selection_selected_score_mean"] = (
            float(np.mean(selected_scores)) if selected_scores else 0.0
        )
        diag["densify_selection_selected_score_max"] = (
            float(np.max(selected_scores)) if selected_scores else 0.0
        )
        return observations

    def _xy_from_bool(self, keep: np.ndarray) -> np.ndarray:
        ys, xs = np.where(keep)
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        stride = max(1, int(self.params.mask_sample_stride))
        if stride > 1:
            mask = ((xs % stride) == 0) & ((ys % stride) == 0)
            xs = xs[mask]
            ys = ys[mask]
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        max_pixels = int(self.params.mask_max_pixels)
        if max_pixels > 0 and ys.size > max_pixels:
            keep_idx = np.linspace(0, ys.size - 1, num=max_pixels, dtype=np.int64)
            xs = xs[keep_idx]
            ys = ys[keep_idx]
        return np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)

    def _backproject_xy(self, frame_id: int, xy: np.ndarray) -> tuple[np.ndarray, int]:
        if xy.size == 0:
            return np.empty((0,), dtype=np.int64), 0
        depth = self._depth(frame_id)
        pose = self.stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            return np.empty((0,), dtype=np.int64), int(xy.shape[0])
        h, w = depth.shape
        x = xy[:, 0].astype(np.int64)
        y = xy[:, 1].astype(np.int64)
        in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if not np.any(in_bounds):
            return np.empty((0,), dtype=np.int64), int(xy.shape[0])
        x = x[in_bounds]
        y = y[in_bounds]
        z = depth[y, x]
        valid = np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            return np.empty((0,), dtype=np.int64), int(xy.shape[0])
        x = x[valid].astype(np.float32)
        y = y[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        fx = float(self.intrinsics[0, 0])
        fy = float(self.intrinsics[1, 1])
        cx = float(self.intrinsics[0, 2])
        cy = float(self.intrinsics[1, 2])
        cam = np.stack([(x - cx) * z / fx, (y - cy) * z / fy, z, np.ones_like(z)], axis=1)
        world = (pose @ cam.T).T[:, :3].astype(np.float32)
        finite_world = np.isfinite(world).all(axis=1)
        if not np.any(finite_world):
            return np.empty((0,), dtype=np.int64), int(xy.shape[0])
        dist, idx = self.tree.query(world[finite_world], k=1, distance_upper_bound=float(self.params.nn_radius))
        hit = np.isfinite(dist) & (idx < self.scene_points.shape[0])
        return idx[hit].astype(np.int64), int(xy.shape[0])

    def densify_object(self, value: dict) -> ReliableDensifyResult:
        seed_point_ids = np.asarray(value.get("point_ids", []), dtype=np.int64).reshape(-1)
        valid_seed_point_ids = seed_point_ids[
            (seed_point_ids >= 0) & (seed_point_ids < self.scene_points.shape[0])
        ]
        observations = unique_mask_observations(list(value.get("mask_list", [])))
        supported_seed_modes = {"supported", "boundary", "component"}
        valid_seed_modes = {"none", "all", *supported_seed_modes}
        if self.params.seed_keep_mode not in valid_seed_modes:
            raise ValueError(f"Unsupported seed_keep_mode: {self.params.seed_keep_mode}")
        out_points: set[int] = set()
        if self.params.seed_keep_mode == "all":
            out_points.update(int(v) for v in valid_seed_point_ids.tolist())
        seed_support_counts: Counter[int] = Counter()
        diag = {
            "densify_observations_raw": float(len(observations)),
            "densify_observations_after_quality_filter": 0.0,
            "densify_observations_selected": 0.0,
            "densify_observations_used_for_export": 0.0,
            "densify_observations_considered": float(len(observations)),
            "densify_observations_used": 0.0,
            "densify_seed_input_points": float(seed_point_ids.shape[0]),
            "densify_seed_valid_points": float(valid_seed_point_ids.shape[0]),
            "densify_seed_kept_points": float(len(out_points)),
            "densify_seed_min_support_views": float(max(1, int(self.params.seed_min_support_views))),
            "densify_seed_supported_unique_points": 0.0,
            "densify_seed_pixels_total": 0.0,
            "densify_component_count": 0.0,
            "mask_pixels_total": 0.0,
            "mask_pixels_component": 0.0,
            "mask_pixels_after_boundary": 0.0,
            "mask_pixels_kept": 0.0,
            "boundary_removed_pixels": 0.0,
            "seed_distance_removed_pixels": 0.0,
            "densify_backproject_queries": 0.0,
            "densify_backproject_hits": 0.0,
            "tiny_mask_count": 0.0,
            "large_mask_count": 0.0,
            "densify_selection_score_mean": 0.0,
            "densify_selection_score_max": 0.0,
            "densify_selection_selected_score_mean": 0.0,
            "densify_selection_selected_score_max": 0.0,
            "densify_selection_seed_pixels_sum": 0.0,
            "densify_selection_component_pixels_sum": 0.0,
            "densify_selection_distance_pixels_sum": 0.0,
        }
        observations = self._select_observations(observations, seed_point_ids, diag)
        diag["densify_observations_after_quality_filter"] = float(len(observations))
        diag["densify_observations_selected"] = float(len(observations))
        diag["densify_observations_considered"] = float(len(observations))

        for frame_id, mask_id, _ in observations:
            mask = self._mask(int(frame_id))
            mask_bool = mask == int(mask_id)
            mask_area = int(np.count_nonzero(mask_bool))
            diag["mask_pixels_total"] += float(mask_area)
            if mask_area <= 0:
                continue
            diag["tiny_mask_count"] += float(mask_area < 100)
            diag["large_mask_count"] += float(mask_area > 1000)

            projected, projected_point_ids = self._project_points_with_ids(int(frame_id), seed_point_ids)
            if projected.size == 0:
                continue
            inside = mask_bool[projected[:, 1], projected[:, 0]]
            seed_xy = np.unique(projected[inside], axis=0)
            if seed_xy.shape[0] < int(self.params.min_seed_pixels):
                continue
            diag["densify_seed_pixels_total"] += float(seed_xy.shape[0])

            component, component_count = self._component_from_seeds(mask_bool, seed_xy)
            if not np.any(component):
                continue
            diag["densify_component_count"] += float(component_count)
            component_pixels = int(np.count_nonzero(component))
            diag["mask_pixels_component"] += float(component_pixels)

            filtered = component
            erosion = int(self.params.boundary_erosion)
            if erosion > 0 and component_pixels >= int(self.params.small_mask_area):
                kernel = np.ones((3, 3), dtype=np.uint8)
                eroded = cv2.erode(component.astype(np.uint8), kernel, iterations=erosion).astype(bool)
                if np.any(eroded) and np.any(eroded[seed_xy[:, 1], seed_xy[:, 0]]):
                    filtered = eroded
            after_boundary = int(np.count_nonzero(filtered))
            diag["mask_pixels_after_boundary"] += float(after_boundary)
            diag["boundary_removed_pixels"] += float(max(component_pixels - after_boundary, 0))

            if self.params.seed_keep_mode == "component":
                supported_seed_ids = projected_point_ids[component[projected[:, 1], projected[:, 0]]]
                seed_support_counts.update(set(int(v) for v in supported_seed_ids.tolist()))
            elif self.params.seed_keep_mode == "boundary":
                supported_seed_ids = projected_point_ids[filtered[projected[:, 1], projected[:, 0]]]
                seed_support_counts.update(set(int(v) for v in supported_seed_ids.tolist()))

            distance_filtered = self._distance_filter(filtered, seed_xy, float(self.params.seed_distance_px))
            if not np.any(distance_filtered):
                continue
            if self.params.seed_keep_mode == "supported":
                supported_seed_ids = projected_point_ids[
                    distance_filtered[projected[:, 1], projected[:, 0]]
                ]
                if supported_seed_ids.size:
                    seed_support_counts.update(set(int(v) for v in supported_seed_ids.tolist()))
            after_distance = int(np.count_nonzero(distance_filtered))
            diag["seed_distance_removed_pixels"] += float(max(after_boundary - after_distance, 0))
            diag["mask_pixels_kept"] += float(after_distance)

            xy = self._xy_from_bool(distance_filtered)
            hit_ids, query_count = self._backproject_xy(int(frame_id), xy)
            diag["densify_backproject_queries"] += float(query_count)
            diag["densify_backproject_hits"] += float(hit_ids.shape[0])
            if hit_ids.size:
                diag["densify_observations_used"] += 1.0
                diag["densify_observations_used_for_export"] += 1.0
                out_points.update(int(v) for v in hit_ids.tolist())

        if self.params.seed_keep_mode in supported_seed_modes:
            min_support = max(1, int(self.params.seed_min_support_views))
            keep_seed_ids = [pid for pid, count in seed_support_counts.items() if count >= min_support]
            out_points.update(keep_seed_ids)
            diag["densify_seed_supported_unique_points"] = float(len(seed_support_counts))
            diag["densify_seed_kept_points"] = float(len(keep_seed_ids))

        diag["kept_ratio"] = float(diag["mask_pixels_kept"] / max(diag["mask_pixels_total"], 1.0))
        diag["boundary_removed_ratio"] = float(
            diag["boundary_removed_pixels"] / max(diag["mask_pixels_component"], 1.0)
        )
        diag["seed_distance_removed_ratio"] = float(
            diag["seed_distance_removed_pixels"] / max(diag["mask_pixels_after_boundary"], 1.0)
        )
        diag["densify_backproject_hit_rate"] = float(
            diag["densify_backproject_hits"] / max(diag["densify_backproject_queries"], 1.0)
        )
        diag["densify_output_points_before_wta"] = float(len(out_points))
        diag["densify_added_points_before_wta"] = float(
            max(len(out_points) - valid_seed_point_ids.shape[0], 0)
        )
        return ReliableDensifyResult(point_ids=out_points, diagnostics=diag)


def sum_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        out[f"{key}_sum"] = float(sum(values))
        out[f"{key}_mean"] = float(np.mean(values))
    return out
