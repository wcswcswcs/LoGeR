from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .object_memory import ObjectMemory4D
from .reliable_densifier import (
    ReliableDensifier,
    ReliableDensifyParams,
    apply_wta_to_records,
    sum_diagnostics,
)
from .scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def score_export_record(record: dict, export_score_mode: str) -> float:
    point_count = len(record.get("point_ids", []))
    if export_score_mode == "one":
        return 1.0
    if export_score_mode == "area":
        return float(record.get("area_score", record.get("score", point_count)))
    if export_score_mode == "reliability":
        return float(record.get("reliability", record.get("score", point_count)))
    if export_score_mode == "observations":
        return float(record.get("observations", record.get("reliability", 0.0)))
    if export_score_mode == "dense_quality":
        return float(record.get("dense_quality", record.get("reliability", 0.0)))
    if export_score_mode == "selection_quality":
        return float(record.get("selection_quality", record.get("dense_quality", 0.0)))
    raise ValueError(f"Unsupported export score mode: {export_score_mode}")


class ScanNetExporter:
    def __init__(
        self,
        stream: ScanNetStream,
        output_config: str = "stream4d_scannet",
        export_nn_radius: float = 0.05,
        export_support_mode: str = "carrier_uv",
        export_mask_sample_stride: int = 1,
        export_mask_max_pixels: int = 50000,
        export_max_masks_per_object: int = 0,
        export_mask_min_relative_coverage: float = 0.0,
        export_core_nn_radius: float | None = None,
        export_fringe_nn_radius: float | None = None,
        export_fringe_radius: float = 0.05,
        export_fringe_max_ratio: float = 0.35,
        export_point_dilate_radius: float = 0.0,
        export_min_points_per_object: int = 0,
        export_score_mode: str = "one",
        export_enable_wta: bool = False,
        export_wta_score_mode: str = "evidence_density",
        export_wta_min_conflict_owners: int = 2,
        densify_boundary_erosion: int = 1,
        densify_small_mask_area: int = 400,
        densify_seed_distance_px: float = 32.0,
        densify_min_seed_pixels: int = 1,
        densify_enable_wta: bool = True,
        densify_seed_keep_mode: str = "none",
        densify_seed_min_support_views: int = 1,
        densify_mask_selection_mode: str = "coverage",
    ) -> None:
        self.stream = stream
        self.output_config = output_config
        self.export_nn_radius = float(export_nn_radius)
        if export_support_mode not in {
            "carrier_uv",
            "mask_backproject",
            "hybrid",
            "core_fringe",
            "component_densify",
            "reuse_point_ids",
            "point_dilate",
            "reliable_densify",
            "posterior_support",
        }:
            raise ValueError(f"Unsupported export support mode: {export_support_mode}")
        self.export_support_mode = export_support_mode
        self.export_mask_sample_stride = max(1, int(export_mask_sample_stride))
        self.export_mask_max_pixels = int(export_mask_max_pixels)
        self.export_max_masks_per_object = int(export_max_masks_per_object)
        self.export_mask_min_relative_coverage = max(0.0, float(export_mask_min_relative_coverage))
        self.export_core_nn_radius = (
            float(export_nn_radius) if export_core_nn_radius is None else float(export_core_nn_radius)
        )
        self.export_fringe_nn_radius = (
            float(export_nn_radius) if export_fringe_nn_radius is None else float(export_fringe_nn_radius)
        )
        self.export_fringe_radius = max(0.0, float(export_fringe_radius))
        self.export_fringe_max_ratio = max(0.0, float(export_fringe_max_ratio))
        self.export_point_dilate_radius = float(export_point_dilate_radius)
        self.export_min_points_per_object = int(export_min_points_per_object)
        self.densify_boundary_erosion = int(densify_boundary_erosion)
        self.densify_small_mask_area = int(densify_small_mask_area)
        self.densify_seed_distance_px = float(densify_seed_distance_px)
        self.densify_min_seed_pixels = int(densify_min_seed_pixels)
        self.densify_enable_wta = bool(densify_enable_wta)
        if densify_seed_keep_mode not in {"none", "supported", "boundary", "component", "all"}:
            raise ValueError(f"Unsupported densify_seed_keep_mode: {densify_seed_keep_mode}")
        self.densify_seed_keep_mode = densify_seed_keep_mode
        self.densify_seed_min_support_views = max(1, int(densify_seed_min_support_views))
        valid_selection_modes = {
            "coverage",
            "seed_density",
            "component_seed_density",
            "kept_seed_density",
            "coverage_component_density",
            "coverage_kept_density",
            "kept_ratio",
        }
        if densify_mask_selection_mode not in valid_selection_modes:
            raise ValueError(f"Unsupported densify_mask_selection_mode: {densify_mask_selection_mode}")
        self.densify_mask_selection_mode = densify_mask_selection_mode
        if export_score_mode not in {"one", "area", "reliability", "observations", "dense_quality", "selection_quality"}:
            raise ValueError(f"Unsupported export score mode: {export_score_mode}")
        self.export_score_mode = export_score_mode
        self.export_enable_wta = bool(export_enable_wta)
        self.export_wta_min_conflict_owners = max(2, int(export_wta_min_conflict_owners))
        if export_wta_score_mode not in {
            "evidence_quality",
            "evidence_density",
            "observations",
            "carriers",
            "compactness",
        }:
            raise ValueError(f"Unsupported export WTA score mode: {export_wta_score_mode}")
        self.export_wta_score_mode = export_wta_score_mode
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "open3d is required to construct ScanNetExporter. Pure-python helpers in "
                "stream4d.export_scannet remain importable without open3d."
            ) from exc
        self.scene_points = np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)
        if self.scene_points.ndim != 2 or self.scene_points.shape[1] != 3:
            raise RuntimeError(f"Failed to load scene points from {stream.mesh_path}")
        self.tree = cKDTree(self.scene_points)
        self.intrinsics = stream.load_intrinsics()

    def _backproject_xy(
        self,
        frame_id: int,
        xy: np.ndarray,
        nn_radius: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if xy.size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        depth = self.stream.load_depth(frame_id)
        pose = self.stream.load_pose(frame_id)
        if not np.isfinite(pose).all():
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        h, w = depth.shape
        x = xy[:, 0].astype(np.int64)
        y = xy[:, 1].astype(np.int64)
        in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if not np.any(in_bounds):
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        x = x[in_bounds]
        y = y[in_bounds]
        z = depth[y, x]
        valid = np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
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
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        world = world[finite_world]
        radius = self.export_nn_radius if nn_radius is None else float(nn_radius)
        dist, idx = self.tree.query(world, k=1, distance_upper_bound=radius)
        hit = np.isfinite(dist) & (idx < self.scene_points.shape[0])
        return idx[hit].astype(np.int64), dist[hit].astype(np.float32)

    def _backproject_uv(
        self,
        frame_id: int,
        uv_norm: np.ndarray,
        nn_radius: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if uv_norm.size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        depth = self.stream.load_depth(frame_id)
        h, w = depth.shape
        x = np.rint(uv_norm[:, 0] * float(max(w - 1, 1))).astype(np.int64)
        y = np.rint(uv_norm[:, 1] * float(max(h - 1, 1))).astype(np.int64)
        return self._backproject_xy(frame_id, np.stack([x, y], axis=1), nn_radius=nn_radius)

    def _mask_pixels(self, frame_id: int, mask_id: int) -> np.ndarray:
        depth = self.stream.load_depth(frame_id)
        mask = self.stream.load_mask(frame_id)
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask == int(mask_id))
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        stride = self.export_mask_sample_stride
        if stride > 1:
            keep = ((xs % stride) == 0) & ((ys % stride) == 0)
            xs = xs[keep]
            ys = ys[keep]
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        max_pixels = self.export_mask_max_pixels
        if max_pixels > 0 and ys.size > max_pixels:
            keep = np.linspace(0, ys.size - 1, num=max_pixels, dtype=np.int64)
            xs = xs[keep]
            ys = ys[keep]
        return np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)

    def _backproject_mask(
        self,
        frame_id: int,
        mask_id: int,
        nn_radius: float | None = None,
    ) -> tuple[np.ndarray, int]:
        xy = self._mask_pixels(frame_id, mask_id)
        if xy.size == 0:
            return np.empty((0,), dtype=np.int64), 0
        hit_ids, _ = self._backproject_xy(frame_id, xy, nn_radius=nn_radius)
        return hit_ids, int(xy.shape[0])

    def _write_outputs(
        self,
        object_records: list[dict],
        object_dict: dict[int, dict],
        point_owner_counts: np.ndarray,
        *,
        write_manifest: bool = True,
    ) -> dict[str, float]:
        kept_records = [record for record in object_records if len(record["point_ids"]) >= self.export_min_points_per_object]
        kept_ids = {int(record["object_id"]) for record in kept_records}
        object_dict = {int(k): v for k, v in object_dict.items() if int(k) in kept_ids}

        point_owner_counts = np.zeros((self.scene_points.shape[0],), dtype=np.uint16)
        masks = np.zeros((self.scene_points.shape[0], len(kept_records)), dtype=bool)
        scores = np.zeros((len(kept_records),), dtype=np.float32)
        for out_idx, record in enumerate(kept_records):
            point_ids = record["point_ids"]
            if point_ids:
                ids = np.fromiter(point_ids, dtype=np.int64)
                masks[ids, out_idx] = True
                point_owner_counts[ids] += 1
            scores[out_idx] = float(score_export_record(record, self.export_score_mode))

        pred_dir = Path("data/prediction") / f"{self.output_config}_class_agnostic"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_dict = {
            "pred_masks": masks,
            "pred_score": scores,
            "pred_classes": np.zeros((len(kept_records),), dtype=np.int32),
        }
        np.savez_compressed(pred_dir / f"{self.stream.seq_name}.npz", **pred_dict)

        object_dir = self.stream.object_dir / self.output_config
        object_dir.mkdir(parents=True, exist_ok=True)
        np.save(object_dir / "object_dict.npy", object_dict, allow_pickle=True)

        pre_points = np.flatnonzero(point_owner_counts > 0).astype(np.int64)
        tmp_dir = Path("data/TMP") / self.output_config
        tmp_dir.mkdir(parents=True, exist_ok=True)
        np.save(tmp_dir / f"{self.stream.seq_name}_pre_points.npy", pre_points)
        if write_manifest:
            manifest = build_prediction_manifest(
                output_config=self.output_config,
                is_method_result=True,
                is_diagnostic_only=False,
                uses_gt=False,
                gt_usage="none",
                source_configs=[],
                pre_points_policy="recompute",
                support_policy=self.export_support_mode,
                notes="Generated by ScanNetExporter; caller may overwrite with more specific provenance.",
            )
            write_prediction_manifest(self.output_config, manifest)

        conflict_points = int(np.count_nonzero(point_owner_counts > 1))
        return {
            "num_candidate_objects": float(len(object_records)),
            "num_exported_objects": float(len(kept_records)),
            "num_kept_objects": float(len(kept_records)),
            "num_scene_points": float(self.scene_points.shape[0]),
            "num_exported_points": float(pre_points.shape[0]),
            "export_conflict_rate": float(conflict_points / max(pre_points.shape[0], 1)),
        }

    def _dilate_point_ids(self, point_ids: set[int]) -> set[int]:
        radius = self.export_point_dilate_radius
        if radius <= 0.0 or not point_ids:
            return point_ids
        seed_ids = np.fromiter(point_ids, dtype=np.int64)
        seed_ids = seed_ids[(seed_ids >= 0) & (seed_ids < self.scene_points.shape[0])]
        if seed_ids.size == 0:
            return set()
        neighbors = self.tree.query_ball_point(self.scene_points[seed_ids], r=radius)
        expanded = set(int(v) for v in seed_ids.tolist())
        for item in neighbors:
            expanded.update(int(v) for v in item)
        return expanded

    def _empty_diag(self) -> dict[str, float]:
        return {
            "export_carrier_queries": 0.0,
            "export_carrier_hits": 0.0,
            "export_mask_queries": 0.0,
            "export_mask_hits": 0.0,
            "export_num_mask_observations": 0.0,
            "export_core_fringe_core_points": 0.0,
            "export_core_fringe_candidate_points": 0.0,
            "export_core_fringe_kept_points": 0.0,
            "export_core_fringe_objects_with_fringe": 0.0,
        }

    @staticmethod
    def _unique_mask_observations(mask_observations: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        best: dict[tuple[int, int], float] = {}
        for frame_id, mask_id, coverage in mask_observations:
            key = (int(frame_id), int(mask_id))
            best[key] = max(float(coverage), best.get(key, 0.0))
        out = [(frame_id, mask_id, coverage) for (frame_id, mask_id), coverage in best.items()]
        return sorted(out, key=lambda item: float(item[2]), reverse=True)

    def _add_mask_points(
        self,
        point_ids: set[int],
        mask_observations: list[tuple[int, int, float]],
        diag: dict[str, float],
        nn_radius: float | None = None,
    ) -> None:
        unique_observations = self._unique_mask_observations(mask_observations)
        if unique_observations and self.export_mask_min_relative_coverage > 0.0:
            top_coverage = max(float(unique_observations[0][2]), 1e-12)
            unique_observations = [
                item
                for item in unique_observations
                if float(item[2]) >= top_coverage * self.export_mask_min_relative_coverage
            ]
        if self.export_max_masks_per_object > 0:
            unique_observations = unique_observations[: self.export_max_masks_per_object]
        for frame_id, mask_id, _ in unique_observations:
            hit_ids, query_count = self._backproject_mask(int(frame_id), int(mask_id), nn_radius=nn_radius)
            diag["export_mask_queries"] += float(query_count)
            diag["export_mask_hits"] += float(hit_ids.shape[0])
            diag["export_num_mask_observations"] += 1.0
            point_ids.update(int(v) for v in hit_ids.tolist())

    def _add_carrier_points(
        self,
        point_ids: set[int],
        frame_support: dict[int, set[int]],
        support_uv: dict[tuple[int, int], tuple[np.ndarray, float]],
        diag: dict[str, float],
        nn_radius: float | None = None,
    ) -> None:
        for frame_id, carrier_ids in frame_support.items():
            uv_list = []
            for carrier_id in carrier_ids:
                item = support_uv.get((int(frame_id), int(carrier_id)))
                if item is None:
                    continue
                uv_list.append(item[0])
            if not uv_list:
                continue
            uv = np.stack(uv_list, axis=0).astype(np.float32)
            diag["export_carrier_queries"] += float(uv.shape[0])
            hit_ids, _ = self._backproject_uv(int(frame_id), uv, nn_radius=nn_radius)
            diag["export_carrier_hits"] += float(hit_ids.shape[0])
            point_ids.update(int(v) for v in hit_ids.tolist())

    def _select_core_fringe_points(
        self,
        core_points: set[int],
        fringe_candidates: set[int],
        diag: dict[str, float],
    ) -> set[int]:
        diag["export_core_fringe_core_points"] += float(len(core_points))
        diag["export_core_fringe_candidate_points"] += float(len(fringe_candidates))
        if not core_points:
            return set()
        if not fringe_candidates or self.export_fringe_radius <= 0.0:
            return set(core_points)

        core_ids = np.fromiter(core_points, dtype=np.int64)
        core_ids = core_ids[(core_ids >= 0) & (core_ids < self.scene_points.shape[0])]
        if core_ids.size == 0:
            return set()

        candidate_ids = np.asarray(
            sorted(int(v) for v in fringe_candidates.difference(core_points)),
            dtype=np.int64,
        )
        candidate_ids = candidate_ids[(candidate_ids >= 0) & (candidate_ids < self.scene_points.shape[0])]
        if candidate_ids.size == 0:
            return set(int(v) for v in core_ids.tolist())

        core_tree = cKDTree(self.scene_points[core_ids])
        distances, _ = core_tree.query(
            self.scene_points[candidate_ids],
            k=1,
            distance_upper_bound=self.export_fringe_radius,
        )
        valid = np.isfinite(distances) & (distances <= self.export_fringe_radius)
        kept_ids = candidate_ids[valid]
        kept_distances = distances[valid]
        if kept_ids.size and self.export_fringe_max_ratio > 0.0:
            max_keep = int(np.ceil(float(core_ids.size) * self.export_fringe_max_ratio))
            if max_keep >= 0 and kept_ids.size > max_keep:
                order = np.argsort(kept_distances, kind="mergesort")[:max_keep]
                kept_ids = kept_ids[order]

        diag["export_core_fringe_kept_points"] += float(kept_ids.size)
        if kept_ids.size:
            diag["export_core_fringe_objects_with_fringe"] += 1.0
        out = set(int(v) for v in core_ids.tolist())
        out.update(int(v) for v in kept_ids.tolist())
        return out

    def _finalize_diag(self, output_diag: dict[str, float], diag: dict[str, float]) -> dict[str, float]:
        out = {
            **output_diag,
            **diag,
            "export_support_mode": self.export_support_mode,
            "export_mask_sample_stride": float(self.export_mask_sample_stride),
            "export_mask_max_pixels": float(self.export_mask_max_pixels),
            "export_max_masks_per_object": float(self.export_max_masks_per_object),
            "export_mask_min_relative_coverage": float(self.export_mask_min_relative_coverage),
            "export_core_nn_radius": float(self.export_core_nn_radius),
            "export_fringe_nn_radius": float(self.export_fringe_nn_radius),
            "export_fringe_radius": float(self.export_fringe_radius),
            "export_fringe_max_ratio": float(self.export_fringe_max_ratio),
            "export_point_dilate_radius": float(self.export_point_dilate_radius),
            "export_min_points_per_object": float(self.export_min_points_per_object),
            "export_score_mode": self.export_score_mode,
            "export_enable_wta": float(self.export_enable_wta),
            "export_wta_score_mode": self.export_wta_score_mode,
            "export_wta_min_conflict_owners": float(self.export_wta_min_conflict_owners),
        }
        total_queries = diag["export_carrier_queries"] + diag["export_mask_queries"]
        total_hits = diag["export_carrier_hits"] + diag["export_mask_hits"]
        if self.export_support_mode in {"reuse_point_ids", "point_dilate"}:
            out["reuse_point_count"] = float(diag["export_carrier_queries"])
            out["reuse_point_after_dilation_count"] = float(diag["export_carrier_hits"])
            out["export_nn_hit_rate"] = None
            out["export_reuse_point_expansion_rate"] = float(
                diag["export_carrier_hits"] / max(diag["export_carrier_queries"], 1.0)
            )
        else:
            out["reuse_point_count"] = None
            out["reuse_point_after_dilation_count"] = None
            out["export_nn_hit_rate"] = float(total_hits / max(total_queries, 1.0))
            out["export_reuse_point_expansion_rate"] = None
        out["export_carrier_hit_rate"] = float(diag["export_carrier_hits"] / max(diag["export_carrier_queries"], 1.0))
        out["export_mask_hit_rate"] = float(diag["export_mask_hits"] / max(diag["export_mask_queries"], 1.0))
        return out

    def export_object_dict_mask_backproject(self, input_object_dict: dict[int, dict]) -> dict[str, float]:
        object_records: list[dict] = []
        output_object_dict: dict[int, dict] = {}
        point_owner_counts = np.zeros((self.scene_points.shape[0],), dtype=np.uint16)
        diag = self._empty_diag()

        for out_idx, (object_id, value) in enumerate(sorted(input_object_dict.items(), key=lambda item: int(item[0]))):
            mask_observations = list(value.get("mask_list", []))
            point_ids: set[int] = set()
            self._add_mask_points(point_ids, mask_observations, diag)
            point_ids = self._dilate_point_ids(point_ids)
            if point_ids:
                ids = np.fromiter(point_ids, dtype=np.int64)
                point_owner_counts[ids] += 1
            carrier_ids = value.get("carrier_ids", np.empty((0,), dtype=np.int64))
            output_object_dict[int(object_id)] = {
                "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
                "mask_list": mask_observations,
                "repre_mask_list": self._representative_masks(mask_observations),
                "carrier_ids": np.asarray(carrier_ids, dtype=np.int64),
            }
            unique_observations: set[tuple[int, int]] = set()
            for item in mask_observations:
                if isinstance(item, dict):
                    frame_id = int(item.get("frame_id", -1))
                    mask_id = int(item.get("mask_id", -1))
                else:
                    frame_id = int(item[0])
                    mask_id = int(item[1])
                unique_observations.add((frame_id, mask_id))
            area_score = float(len(point_ids))
            observation_score = float(len(unique_observations))
            object_records.append(
                {
                    "object_id": int(object_id),
                    "point_ids": point_ids,
                    "score": area_score,
                    "area_score": area_score,
                    "observations": observation_score,
                    "carrier_count": float(np.asarray(carrier_ids).shape[0]),
                    "reliability": float(observation_score * math.sqrt(max(area_score, 1.0))),
                }
            )

        output_diag = self._write_outputs(
            object_records,
            output_object_dict,
            point_owner_counts,
            write_manifest=False,
        )
        manifest = build_prediction_manifest(
            output_config=self.output_config,
            is_method_result=False,
            is_diagnostic_only=True,
            uses_gt=False,
            gt_usage="scannet_rgbd_pose_mesh_mask_backproject",
            source_configs=[],
            pre_points_policy="diagnostic_mask_backproject",
            support_policy=self.export_support_mode,
            notes=(
                "Diagnostic-only mask backproject export. This path uses ScanNet RGB-D/pose/mesh "
                "to materialize evaluator masks and must not enter native method tables."
            ),
            extra={
                "uses_gt_for_prediction": False,
                "uses_rgbd_for_prediction": False,
                "uses_pose_for_prediction": False,
                "uses_scannet_mesh_for_prediction": False,
                "uses_rgbd_for_evaluation": True,
                "uses_rgbd_for_evaluation_support": True,
                "uses_rgbd_pose_mesh_for_export": True,
                "uses_gt_for_diagnostic": True,
                "forbidden_for_method_table": True,
                "geometry_source": "scannet_rgbd_pose_mesh_mask_backproject_eval_adapter",
                "alignment_source": "none",
                "eval_policy": "diagnostic_mask_backproject_only",
                "is_method_result": False,
                "is_diagnostic_only": True,
            },
        )
        write_prediction_manifest(self.output_config, manifest)
        return self._finalize_diag(output_diag, diag)

    def export_object_dict_points(self, input_object_dict: dict[int, dict]) -> dict[str, float]:
        object_records: list[dict] = []
        output_object_dict: dict[int, dict] = {}
        point_owner_counts = np.zeros((self.scene_points.shape[0],), dtype=np.uint16)
        diag = self._empty_diag()

        for object_id, value in sorted(input_object_dict.items(), key=lambda item: int(item[0])):
            raw_point_ids = np.asarray(value.get("point_ids", []), dtype=np.int64)
            point_ids = set(int(v) for v in raw_point_ids.tolist())
            diag["export_carrier_queries"] += float(len(point_ids))
            point_ids = self._dilate_point_ids(point_ids)
            diag["export_carrier_hits"] += float(len(point_ids))
            if point_ids:
                ids = np.fromiter(point_ids, dtype=np.int64)
                point_owner_counts[ids] += 1
            mask_observations = list(value.get("mask_list", []))
            carrier_ids = value.get("carrier_ids", np.empty((0,), dtype=np.int64))
            output_object_dict[int(object_id)] = {
                "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
                "mask_list": mask_observations,
                "repre_mask_list": self._representative_masks(mask_observations),
                "carrier_ids": np.asarray(carrier_ids, dtype=np.int64),
            }
            object_records.append({"object_id": int(object_id), "point_ids": point_ids, "score": float(len(point_ids))})

        output_diag = self._write_outputs(object_records, output_object_dict, point_owner_counts)
        return self._finalize_diag(output_diag, diag)

    def export_object_slot_posterior_support(self, input_object_dict: dict[int, dict], bank: object) -> dict[str, float]:
        object_records: list[dict] = []
        output_object_dict: dict[int, dict] = {}
        point_owner_counts = np.zeros((self.scene_points.shape[0],), dtype=np.uint16)
        diag = self._empty_diag()
        posterior_diag = {
            "posterior_core_surfels": 0.0,
            "posterior_fringe_surfels": 0.0,
            "posterior_unknown_surfels_not_exported": 0.0,
            "posterior_reject_surfels_not_exported": 0.0,
            "posterior_core_exported_points": 0.0,
            "posterior_fringe_candidate_points": 0.0,
            "posterior_fringe_kept_points": 0.0,
            "posterior_connected_component_count": 0.0,
        }

        uv_pred = np.asarray(getattr(bank, "uv_pred"))
        visible_ok = np.asarray(getattr(bank, "visible_ok"), dtype=bool)
        frame_ids = np.asarray(getattr(bank, "frame_ids"), dtype=np.int64)

        def add_surfel_uv_points(surfels: np.ndarray, nn_radius: float) -> set[int]:
            points: set[int] = set()
            surfels = np.asarray(surfels, dtype=np.int64)
            if surfels.size == 0:
                return points
            for frame_idx, frame_id in enumerate(frame_ids.tolist()):
                valid = visible_ok[frame_idx, surfels]
                frame_surfels = surfels[valid]
                if frame_surfels.size == 0:
                    continue
                uv = uv_pred[frame_idx, frame_surfels]
                diag["export_carrier_queries"] += float(uv.shape[0])
                hit_ids, _ = self._backproject_uv(int(frame_id), uv, nn_radius=nn_radius)
                diag["export_carrier_hits"] += float(hit_ids.shape[0])
                points.update(int(v) for v in hit_ids.tolist())
            return points

        for object_id, value in sorted(input_object_dict.items(), key=lambda item: int(item[0])):
            core_surfels = np.asarray(value.get("core_surfels", value.get("carrier_ids", [])), dtype=np.int64)
            fringe_surfels = np.asarray(value.get("fringe_surfels", []), dtype=np.int64)
            unknown_surfels = np.asarray(value.get("unknown_surfels", []), dtype=np.int64)
            reject_surfels = np.asarray(value.get("reject_surfels", []), dtype=np.int64)
            posterior_diag["posterior_core_surfels"] += float(core_surfels.shape[0])
            posterior_diag["posterior_fringe_surfels"] += float(fringe_surfels.shape[0])
            posterior_diag["posterior_unknown_surfels_not_exported"] += float(unknown_surfels.shape[0])
            posterior_diag["posterior_reject_surfels_not_exported"] += float(reject_surfels.shape[0])

            core_points = add_surfel_uv_points(core_surfels, self.export_core_nn_radius)
            fringe_candidates = add_surfel_uv_points(fringe_surfels, self.export_fringe_nn_radius)
            posterior_diag["posterior_core_exported_points"] += float(len(core_points))
            posterior_diag["posterior_fringe_candidate_points"] += float(len(fringe_candidates))
            selected = self._select_core_fringe_points(core_points, fringe_candidates, diag)
            posterior_diag["posterior_fringe_kept_points"] += float(max(len(selected) - len(core_points), 0))
            if selected:
                posterior_diag["posterior_connected_component_count"] += 1.0
            selected = self._dilate_point_ids(selected)
            if selected:
                ids = np.fromiter(selected, dtype=np.int64)
                point_owner_counts[ids] += 1
            mask_observations = list(value.get("mask_list", []))
            output_object_dict[int(object_id)] = {
                "point_ids": np.asarray(sorted(selected), dtype=np.int64),
                "mask_list": mask_observations,
                "repre_mask_list": self._representative_masks(mask_observations),
                "carrier_ids": np.asarray(core_surfels, dtype=np.int64),
                "core_surfels": np.asarray(core_surfels, dtype=np.int64),
                "fringe_surfels": np.asarray(fringe_surfels, dtype=np.int64),
                "unknown_surfels": np.asarray(unknown_surfels, dtype=np.int64),
                "reject_surfels": np.asarray(reject_surfels, dtype=np.int64),
            }
            object_records.append(
                {
                    "object_id": int(object_id),
                    "point_ids": selected,
                    "area_score": float(len(selected)),
                    "score": float(len(selected)),
                    "observations": float(len(self._unique_mask_observations(mask_observations))),
                    "carrier_count": float(core_surfels.shape[0]),
                    "reliability": float(len(self._unique_mask_observations(mask_observations)) * math.sqrt(max(len(selected), 1))),
                }
            )

        wta_diag: dict[str, float] = {}
        if self.export_enable_wta:
            object_records, wta_diag = apply_wta_to_records(object_records)
            for record in object_records:
                object_id = int(record["object_id"])
                if object_id in output_object_dict:
                    output_object_dict[object_id]["point_ids"] = np.asarray(
                        sorted(record["point_ids"]),
                        dtype=np.int64,
                    )

        output_diag = self._write_outputs(object_records, output_object_dict, point_owner_counts)
        output_diag.update(posterior_diag)
        output_diag.update(wta_diag)
        return self._finalize_diag(output_diag, diag)

    def export_object_dict_reliable_densify(self, input_object_dict: dict[int, dict]) -> dict[str, float]:
        params = ReliableDensifyParams(
            max_masks_per_object=self.export_max_masks_per_object,
            mask_min_relative_coverage=self.export_mask_min_relative_coverage,
            mask_sample_stride=self.export_mask_sample_stride,
            mask_max_pixels=self.export_mask_max_pixels,
            boundary_erosion=self.densify_boundary_erosion,
            small_mask_area=self.densify_small_mask_area,
            seed_distance_px=self.densify_seed_distance_px,
            min_seed_pixels=self.densify_min_seed_pixels,
            nn_radius=self.export_nn_radius,
            seed_keep_mode=self.densify_seed_keep_mode,
            seed_min_support_views=self.densify_seed_min_support_views,
            mask_selection_mode=self.densify_mask_selection_mode,
        )
        densifier = ReliableDensifier(
            self.stream,
            scene_points=self.scene_points,
            tree=self.tree,
            intrinsics=self.intrinsics,
            params=params,
        )
        object_records: list[dict] = []
        output_object_dict: dict[int, dict] = {}
        object_diags: list[dict[str, float]] = []
        diag = self._empty_diag()

        for object_id, value in sorted(input_object_dict.items(), key=lambda item: int(item[0])):
            result = densifier.densify_object(value)
            object_diags.append(result.diagnostics)
            mask_observations = list(value.get("mask_list", []))
            carrier_ids = value.get("carrier_ids", np.empty((0,), dtype=np.int64))
            output_object_dict[int(object_id)] = {
                "point_ids": np.asarray(sorted(result.point_ids), dtype=np.int64),
                "mask_list": mask_observations,
                "repre_mask_list": self._representative_masks(mask_observations),
                "carrier_ids": np.asarray(carrier_ids, dtype=np.int64),
            }
            reliability = float(result.diagnostics.get("densify_observations_used", 0.0)) * np.sqrt(
                max(float(len(result.point_ids)), 1.0)
            )
            observations_used = float(result.diagnostics.get("densify_observations_used", 0.0))
            dense_hits = float(result.diagnostics.get("densify_backproject_hits", 0.0))
            dense_ratio = float(dense_hits / max(float(len(result.point_ids)), 1.0))
            dense_quality = observations_used * np.sqrt(max(dense_hits, 1.0)) * min(dense_ratio, 1.0)
            selected_score = max(float(result.diagnostics.get("densify_selection_selected_score_mean", 0.0)), 0.0)
            selection_quality = selected_score * observations_used * np.sqrt(max(float(len(result.point_ids)), 1.0))
            object_records.append(
                {
                    "object_id": int(object_id),
                    "point_ids": set(result.point_ids),
                    "area_score": float(len(result.point_ids)),
                    "score": float(len(result.point_ids)),
                    "reliability": reliability,
                    "observations": observations_used,
                    "dense_quality": float(dense_quality),
                    "selection_quality": float(selection_quality),
                }
            )

        wta_diag: dict[str, float] = {}
        if self.densify_enable_wta:
            object_records, wta_diag = apply_wta_to_records(object_records)
            for record in object_records:
                object_id = int(record["object_id"])
                if object_id in output_object_dict:
                    output_object_dict[object_id]["point_ids"] = np.asarray(
                        sorted(record["point_ids"]),
                        dtype=np.int64,
                    )

        output_diag = self._write_outputs(
            object_records,
            output_object_dict,
            np.zeros((self.scene_points.shape[0],), dtype=np.uint16),
        )
        dense_diag = sum_diagnostics(object_diags)
        diag["export_mask_queries"] = float(dense_diag.get("densify_backproject_queries_sum", 0.0))
        diag["export_mask_hits"] = float(dense_diag.get("densify_backproject_hits_sum", 0.0))
        diag["export_num_mask_observations"] = float(dense_diag.get("densify_observations_used_sum", 0.0))
        output_diag.update(
            {
                **dense_diag,
                **wta_diag,
                "densify_boundary_erosion": float(self.densify_boundary_erosion),
                "densify_small_mask_area": float(self.densify_small_mask_area),
                "densify_seed_distance_px": float(self.densify_seed_distance_px),
                "densify_min_seed_pixels": float(self.densify_min_seed_pixels),
                "densify_enable_wta": float(self.densify_enable_wta),
                "densify_seed_keep_mode": self.densify_seed_keep_mode,
                "densify_seed_min_support_views": float(self.densify_seed_min_support_views),
                "densify_mask_selection_mode": self.densify_mask_selection_mode,
            }
        )
        return self._finalize_diag(output_diag, diag)

    def _object_record_scores(self, obj: object, point_ids: set[int]) -> dict[str, float]:
        observations = float(len(self._unique_mask_observations(list(getattr(obj, "mask_observations", [])))))
        carrier_count = float(len(getattr(obj, "carrier_ids", [])))
        frame_count = float(len(getattr(obj, "frame_support", {})))
        evidence_mean_coverage = float(getattr(obj, "evidence_mean_coverage", 0.0))
        if evidence_mean_coverage <= 0.0:
            coverages = [float(item[2]) for item in getattr(obj, "mask_observations", [])]
            evidence_mean_coverage = float(np.mean(coverages)) if coverages else 0.0
        evidence_quality = float(
            getattr(
                obj,
                "evidence_quality",
                observations
                * np.sqrt(max(carrier_count, 1.0))
                * np.sqrt(max(frame_count, 1.0))
                * max(evidence_mean_coverage, 1e-6),
            )
        )
        area = float(len(point_ids))
        compactness = 0.0
        if point_ids:
            ids = np.fromiter(point_ids, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < self.scene_points.shape[0])]
            if ids.size:
                pts = self.scene_points[ids]
                extent = np.ptp(pts, axis=0)
                diag = float(np.linalg.norm(extent))
                compactness = float(np.sqrt(max(area, 1.0)) / max(diag, 1e-3))
        evidence_density = float(evidence_quality / np.sqrt(max(area, 1.0)))
        if self.export_wta_score_mode == "evidence_quality":
            reliability = evidence_quality
        elif self.export_wta_score_mode == "evidence_density":
            reliability = evidence_density
        elif self.export_wta_score_mode == "observations":
            reliability = observations
        elif self.export_wta_score_mode == "carriers":
            reliability = carrier_count
        elif self.export_wta_score_mode == "compactness":
            reliability = compactness
        else:
            raise ValueError(f"Unsupported export WTA score mode: {self.export_wta_score_mode}")
        return {
            "area_score": area,
            "score": area,
            "reliability": float(reliability),
            "observations": observations,
            "carrier_count": carrier_count,
            "frame_count": frame_count,
            "evidence_quality": evidence_quality,
            "evidence_density": evidence_density,
            "compactness": compactness,
        }

    def export_rgbd_eval(
        self,
        memory: ObjectMemory4D,
        support_uv: dict[tuple[int, int], tuple[np.ndarray, float]],
    ) -> dict[str, float]:
        objects = [obj for obj in sorted(memory.objects.values(), key=lambda item: item.object_id) if obj.carrier_ids]
        object_dict: dict[int, dict] = {}
        object_records: list[dict] = []
        point_owner_counts = np.zeros((self.scene_points.shape[0],), dtype=np.uint16)
        diag = self._empty_diag()
        component_densifier: ReliableDensifier | None = None
        component_densify_diags: list[dict[str, float]] = []
        if self.export_support_mode == "component_densify":
            component_densifier = ReliableDensifier(
                self.stream,
                scene_points=self.scene_points,
                tree=self.tree,
                intrinsics=self.intrinsics,
                params=ReliableDensifyParams(
                    max_masks_per_object=self.export_max_masks_per_object,
                    mask_min_relative_coverage=self.export_mask_min_relative_coverage,
                    mask_sample_stride=self.export_mask_sample_stride,
                    mask_max_pixels=self.export_mask_max_pixels,
                    boundary_erosion=self.densify_boundary_erosion,
                    small_mask_area=self.densify_small_mask_area,
                    seed_distance_px=self.densify_seed_distance_px,
                    min_seed_pixels=self.densify_min_seed_pixels,
                    nn_radius=self.export_core_nn_radius,
                    seed_keep_mode=self.densify_seed_keep_mode,
                    seed_min_support_views=self.densify_seed_min_support_views,
                    mask_selection_mode=self.densify_mask_selection_mode,
                ),
            )

        for obj in objects:
            point_ids: set[int] = set()
            if self.export_support_mode == "core_fringe":
                core_points: set[int] = set()
                fringe_candidates: set[int] = set()
                self._add_mask_points(
                    core_points,
                    obj.mask_observations,
                    diag,
                    nn_radius=self.export_core_nn_radius,
                )
                self._add_carrier_points(
                    fringe_candidates,
                    obj.frame_support,
                    support_uv,
                    diag,
                    nn_radius=self.export_fringe_nn_radius,
                )
                point_ids = self._select_core_fringe_points(core_points, fringe_candidates, diag)
            elif self.export_support_mode == "component_densify":
                if component_densifier is None:
                    raise RuntimeError("component_densify exporter was not initialized")
                seed_points: set[int] = set()
                self._add_carrier_points(
                    seed_points,
                    obj.frame_support,
                    support_uv,
                    diag,
                    nn_radius=self.export_fringe_nn_radius,
                )
                result = component_densifier.densify_object(
                    {
                        "point_ids": np.asarray(sorted(seed_points), dtype=np.int64),
                        "mask_list": list(obj.mask_observations),
                        "carrier_ids": np.asarray(sorted(obj.carrier_ids), dtype=np.int64),
                    }
                )
                component_densify_diags.append(result.diagnostics)
                point_ids = set(result.point_ids)
            elif self.export_support_mode in {"carrier_uv", "hybrid"}:
                self._add_carrier_points(point_ids, obj.frame_support, support_uv, diag)
            if self.export_support_mode in {"mask_backproject", "hybrid"}:
                self._add_mask_points(point_ids, obj.mask_observations, diag)
            point_ids = self._dilate_point_ids(point_ids)
            if point_ids:
                ids = np.fromiter(point_ids, dtype=np.int64)
                point_owner_counts[ids] += 1
            object_dict[int(obj.object_id)] = {
                "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
                "mask_list": list(obj.mask_observations),
                "repre_mask_list": self._representative_masks(obj.mask_observations),
                "carrier_ids": np.asarray(sorted(obj.carrier_ids), dtype=np.int64),
            }
            object_records.append(
                {
                    "object_id": int(obj.object_id),
                    "point_ids": point_ids,
                    **self._object_record_scores(obj, point_ids),
                    "selection_quality": float(
                        max(
                            float(
                                component_densify_diags[-1].get(
                                    "densify_selection_selected_score_mean",
                                    0.0,
                                )
                            ),
                            0.0,
                        )
                        * float(component_densify_diags[-1].get("densify_observations_used", 0.0))
                        * np.sqrt(max(float(len(point_ids)), 1.0))
                    )
                    if self.export_support_mode == "component_densify" and component_densify_diags
                    else 0.0,
                }
            )

        wta_diag: dict[str, float] = {}
        if self.export_enable_wta:
            object_records, wta_diag = apply_wta_to_records(
                object_records,
                min_conflict_owners=self.export_wta_min_conflict_owners,
            )
            for record in object_records:
                object_id = int(record["object_id"])
                if object_id in object_dict:
                    object_dict[object_id]["point_ids"] = np.asarray(
                        sorted(record["point_ids"]),
                        dtype=np.int64,
                    )

        output_diag = self._write_outputs(
            object_records,
            object_dict,
            point_owner_counts,
            write_manifest=False,
        )
        manifest = build_prediction_manifest(
            output_config=self.output_config,
            is_method_result=False,
            is_diagnostic_only=True,
            uses_gt=False,
            gt_usage="none",
            source_configs=[],
            pre_points_policy="diagnostic_recompute",
            support_policy=self.export_support_mode,
            notes=(
                "Diagnostic-only RGB-D bridge export. This path uses ScanNet RGB-D/pose/mesh "
                "to materialize support and must not enter native method tables."
            ),
            extra={
                "uses_rgbd_for_prediction": True,
                "uses_pose_for_prediction": True,
                "uses_scannet_mesh_for_prediction": True,
                "uses_gt_for_prediction": False,
                "uses_gt_sim3_for_prediction": False,
                "uses_rgbd_for_evaluation": True,
                "forbidden_for_method_table": True,
                "geometry_source": "rgbd_eval_bridge",
                "alignment_source": "scannet_pose_depth_mesh_bridge",
                "eval_policy": "diagnostic_rgbd_bridge_only",
                "is_method_result": False,
                "is_diagnostic_only": True,
            },
        )
        write_prediction_manifest(self.output_config, manifest)
        if component_densify_diags:
            dense_diag = sum_diagnostics(component_densify_diags)
            diag["export_mask_queries"] = float(dense_diag.get("densify_backproject_queries_sum", 0.0))
            diag["export_mask_hits"] = float(dense_diag.get("densify_backproject_hits_sum", 0.0))
            diag["export_num_mask_observations"] = float(dense_diag.get("densify_observations_used_sum", 0.0))
            output_diag.update(
                {
                    **dense_diag,
                    "densify_boundary_erosion": float(self.densify_boundary_erosion),
                    "densify_small_mask_area": float(self.densify_small_mask_area),
                    "densify_seed_distance_px": float(self.densify_seed_distance_px),
                    "densify_min_seed_pixels": float(self.densify_min_seed_pixels),
                    "densify_seed_keep_mode": self.densify_seed_keep_mode,
                    "densify_seed_min_support_views": float(self.densify_seed_min_support_views),
                    "densify_mask_selection_mode": self.densify_mask_selection_mode,
                }
            )
        output_diag.update(wta_diag)
        return self._finalize_diag(output_diag, diag)

    @staticmethod
    def _representative_masks(mask_observations: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        ranked = sorted(mask_observations, key=lambda item: float(item[2]), reverse=True)
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int, float]] = []
        for frame_id, mask_id, coverage in ranked:
            key = (int(frame_id), int(mask_id))
            if key in seen:
                continue
            seen.add(key)
            out.append((int(frame_id), int(mask_id), float(coverage)))
            if len(out) >= 5:
                break
        return out

    def export_d4rt_nn(self, memory: ObjectMemory4D) -> dict[str, float]:
        raise NotImplementedError(
            "d4rt_nn export requires a scene-coordinate calibration path; rgbd_eval is the MVP default."
        )
