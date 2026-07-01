from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.self_stitch import fit_sim3_with_diagnostics, match_overlap_carriers
from stream4d_native.sim3 import fit_sim3_umeyama

from .base import FrameProjection
from .common import backproject_xy_world, fit_transform


def _as_numpy_points(points: Any) -> np.ndarray:
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _apply_fit(points: np.ndarray, fit: dict[str, Any] | None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if fit is None:
        return points
    scale = float(fit["scale"])
    rotation = np.asarray(fit["rotation"], dtype=np.float64)
    translation = np.asarray(fit["translation"], dtype=np.float64)
    return (scale * (points.astype(np.float64) @ rotation.T) + translation).astype(np.float32)


def _fit_summary(fit: dict[str, Any] | None) -> dict[str, Any]:
    if fit is None:
        return {
            "anchor_count": 0,
            "sim3_scale": None,
            "sim3_residual_p90": None,
            "sim3_residual_median": None,
        }
    residual = np.asarray(fit["residual"], dtype=np.float64)
    return {
        "anchor_count": int(fit.get("anchor_count", residual.shape[0])),
        "sim3_scale": float(fit["scale"]),
        "sim3_residual_p90": float(np.percentile(residual, 90)) if residual.size else None,
        "sim3_residual_median": float(np.median(residual)) if residual.size else None,
    }


def _fit_with_fixed_scale(source: np.ndarray, target: np.ndarray, template_fit: dict[str, Any], scale: float) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rotation = np.asarray(template_fit["rotation"], dtype=np.float64)
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    translation = mu_target - float(scale) * (rotation @ mu_source)
    transformed = float(scale) * (source @ rotation.T) + translation
    residual = np.linalg.norm(transformed - target, axis=1)
    return {
        "scale": float(scale),
        "rotation": rotation.astype(np.float64),
        "rotation_det": float(np.linalg.det(rotation)),
        "translation": translation.astype(np.float64),
        "residual": residual.astype(np.float64),
        "anchor_count": int(source.shape[0]),
    }


def _compose_fits(prev_fit: dict[str, Any] | None, pair_fit: dict[str, Any]) -> dict[str, Any]:
    if prev_fit is None:
        return pair_fit
    prev_scale = float(prev_fit["scale"])
    prev_rotation = np.asarray(prev_fit["rotation"], dtype=np.float64)
    prev_translation = np.asarray(prev_fit["translation"], dtype=np.float64)
    pair_scale = float(pair_fit["scale"])
    pair_rotation = np.asarray(pair_fit["rotation"], dtype=np.float64)
    pair_translation = np.asarray(pair_fit["translation"], dtype=np.float64)
    scale = prev_scale * pair_scale
    rotation = prev_rotation @ pair_rotation
    translation = prev_scale * (pair_translation @ prev_rotation.T) + prev_translation
    return {
        "scale": float(scale),
        "rotation": rotation.astype(np.float64),
        "rotation_det": float(np.linalg.det(rotation)),
        "translation": translation.astype(np.float64),
        "residual": np.asarray(pair_fit.get("residual", []), dtype=np.float64),
        "anchor_count": int(pair_fit.get("anchor_count", 0)),
    }


def _spacing_q75(points: np.ndarray, max_points: int = 4096) -> float | None:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return None
    if points.shape[0] > max_points:
        keep = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        points = points[keep]
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=2)
    return float(np.percentile(dist[:, 1], 75))


def _mask_interior_distances(mask: np.ndarray, x: np.ndarray, y: np.ndarray, mask_ids: np.ndarray) -> np.ndarray:
    distances = np.zeros(mask_ids.shape[0], dtype=np.float32)
    for mask_id in np.unique(mask_ids):
        if int(mask_id) <= 0:
            continue
        idx = mask_ids == int(mask_id)
        if not np.any(idx):
            continue
        dist_map = distance_transform_edt(mask == int(mask_id))
        distances[idx] = dist_map[y[idx], x[idx]].astype(np.float32)
    return distances


@dataclass
class _WindowData:
    path: Path
    frame_ids: list[int]
    xyz: np.ndarray
    uv: np.ndarray
    valid: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    carrier_id: np.ndarray
    persistent_tube_id: np.ndarray
    src_frame_global: np.ndarray
    src_xy: np.ndarray
    transform: dict[str, Any] | None


class D4RTCarrierProjectionProvider:
    """Project D4RT carrier geometry into Stream3D's internal mask graph.

    The provider returns the same ``mask_id -> ScanNet scene point ids`` shape
    as the RGB-D backprojection path, so downstream Stream3D set-cover,
    manifold refinement, neighbor merging and historical merging are actually
    rerun. Rows using eval Sim3 remain diagnostic-only.
    """

    uses_rgbd_for_prediction = False
    uses_pose_for_prediction = False
    uses_scannet_mesh_for_prediction = True
    is_diagnostic_only = True

    def __init__(
        self,
        *,
        debug_root: str | Path,
        mode: str,
        nn_radius: float = 0.05,
        nn_k: int = 1,
        min_visibility: float = 0.5,
        min_confidence: float = 0.5,
        max_anchors: int = 8000,
        robust_trim_percentile: float = 90.0,
        density_alpha: float = 2.0,
        local_outlier_filter: bool = False,
        min_mask_interior_px: float = 0.0,
        overlap_policy: str = "all_window_union",
        stitch_uv_radius: float = 0.01,
        stitch_max_matches_per_frame: int = 512,
        stitch_fit_trim_percentile: float = 0.0,
    ) -> None:
        if mode not in {
            "raw",
            "self_stitched",
            "self_stitched_density",
            "self_stitched_eval_sim3",
            "self_stitched_eval_sim3_density",
            "self_stitched_scale_normalized",
            "self_stitched_scale_normalized_density",
            "self_stitched_scale_normalized_eval_sim3",
            "self_stitched_scale_normalized_eval_sim3_density",
            "eval_sim3",
            "eval_sim3_density",
        }:
            raise ValueError(f"Unsupported D4RT carrier provider mode: {mode}")
        self.debug_root = Path(debug_root)
        self.mode = mode
        self.name = f"d4rt_carrier_{mode}"
        self.nn_radius = float(nn_radius)
        self.nn_k = max(1, int(nn_k))
        self.min_visibility = float(min_visibility)
        self.min_confidence = float(min_confidence)
        self.max_anchors = int(max_anchors)
        self.robust_trim_percentile = float(robust_trim_percentile)
        self.density_alpha = float(density_alpha)
        self.local_outlier_filter = bool(local_outlier_filter)
        self.min_mask_interior_px = float(min_mask_interior_px)
        self.stitch_uv_radius = float(stitch_uv_radius)
        self.stitch_max_matches_per_frame = int(stitch_max_matches_per_frame)
        self.stitch_fit_trim_percentile = float(stitch_fit_trim_percentile)
        aliases = {"all": "all_window_union"}
        overlap_policy = aliases.get(str(overlap_policy), str(overlap_policy))
        if overlap_policy not in {"all_window_union", "best_confidence", "lowest_residual", "newest_window"}:
            raise ValueError(f"Unsupported overlap_policy: {overlap_policy}")
        self.overlap_policy = overlap_policy
        self.uses_gt_sim3_for_prediction = mode.startswith("eval_sim3") or "eval_sim3" in mode
        self.uses_d4rt_self_sim3 = mode.startswith("self_stitched")
        self.frame_diagnostics: list[dict[str, Any]] = []
        self._scene_cache: dict[str, dict[str, Any]] = {}

    def reset_diagnostics(self) -> None:
        self.frame_diagnostics.clear()

    def aggregate_diagnostics(self) -> dict[str, Any]:
        rows = self.frame_diagnostics
        numeric_keys = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)
            }
        )
        out = {
            "provider": self.name,
            "mode": self.mode,
            "num_projected_frames": int(len(rows)),
            "uses_gt_sim3_for_prediction": bool(self.uses_gt_sim3_for_prediction),
            "uses_d4rt_self_sim3": bool(self.uses_d4rt_self_sim3),
        }
        for key in numeric_keys:
            vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
        return out

    def write_diagnostics(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.aggregate_diagnostics(), "frames": self.frame_diagnostics}
        path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")

    def _frame_ids_for_window(self, carrier_path: Path, data: dict[str, np.ndarray], num_frames: int) -> list[int]:
        manifest = carrier_path.with_name(f"{carrier_path.stem}_manifest.json")
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                for key in ("raw_frame_ids", "frame_indices", "frame_ids"):
                    vals = [int(v) for v in payload.get(key, [])]
                    if len(vals) == num_frames:
                        return vals
            except Exception:
                pass
        src_frame = np.asarray(data.get("src_frame"), dtype=np.int64)
        src_global = np.asarray(data.get("src_frame_global"), dtype=np.int64)
        if src_frame.shape == src_global.shape and src_frame.size:
            out: list[int] = []
            for local_idx in range(num_frames):
                vals = src_global[src_frame == local_idx]
                if vals.size:
                    uniq, counts = np.unique(vals, return_counts=True)
                    out.append(int(uniq[np.argmax(counts)]))
                else:
                    out.append(int(local_idx))
            return out
        return list(range(num_frames))

    def _load_npz(self, carrier_path: Path) -> dict[str, np.ndarray]:
        with np.load(carrier_path) as data:
            return {key: np.asarray(data[key]) for key in data.files}

    def _collect_eval_anchors(
        self,
        stream: ScanNetStream,
        carrier_paths: list[Path],
        frame_ids_by_path: dict[Path, list[int]],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        source_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        total = 0
        valid_total = 0
        for carrier_path in carrier_paths:
            data = self._load_npz(carrier_path)
            xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
            uv = np.asarray(data["uv_pred"], dtype=np.float32)
            valid = np.asarray(data.get("valid", np.ones(xyz.shape[:2], dtype=bool)), dtype=bool)
            visibility = np.asarray(data.get("visibility_prob", np.ones(xyz.shape[:2], dtype=np.float32)), dtype=np.float32)
            confidence = np.asarray(data.get("confidence_prob", np.ones(xyz.shape[:2], dtype=np.float32)), dtype=np.float32)
            frame_ids = frame_ids_by_path[carrier_path]
            for local_idx, frame_id in enumerate(frame_ids):
                h, w = stream.load_depth(int(frame_id)).shape[:2]
                ok = (
                    valid[local_idx]
                    & np.isfinite(xyz[local_idx]).all(axis=1)
                    & np.isfinite(uv[local_idx]).all(axis=1)
                    & (uv[local_idx, :, 0] >= 0.0)
                    & (uv[local_idx, :, 0] <= 1.0)
                    & (uv[local_idx, :, 1] >= 0.0)
                    & (uv[local_idx, :, 1] <= 1.0)
                    & (visibility[local_idx] >= self.min_visibility)
                    & (confidence[local_idx] >= self.min_confidence)
                )
                total += int(ok.shape[0])
                if not np.any(ok):
                    continue
                xy = np.stack(
                    [
                        uv[local_idx, ok, 0] * float(max(w - 1, 1)),
                        uv[local_idx, ok, 1] * float(max(h - 1, 1)),
                    ],
                    axis=1,
                )
                world, world_ok = backproject_xy_world(stream, int(frame_id), xy)
                if not np.any(world_ok):
                    continue
                source_parts.append(xyz[local_idx, ok][world_ok])
                target_parts.append(world[world_ok])
                valid_total += int(np.count_nonzero(world_ok))
        if not source_parts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), {
                "anchor_candidates": int(total),
                "anchor_valid": 0,
            }
        source = np.concatenate(source_parts, axis=0)
        target = np.concatenate(target_parts, axis=0)
        if source.shape[0] > self.max_anchors:
            keep = np.linspace(0, source.shape[0] - 1, num=self.max_anchors, dtype=np.int64)
            source = source[keep]
            target = target[keep]
        return source, target, {"anchor_candidates": int(total), "anchor_valid": int(valid_total)}

    def _collect_eval_anchors_from_windows(
        self,
        stream: ScanNetStream,
        windows: list[_WindowData],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        source_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        total = 0
        valid_total = 0
        for window in windows:
            for local_idx, frame_id in enumerate(window.frame_ids):
                h, w = stream.load_depth(int(frame_id)).shape[:2]
                xyz = _apply_fit(window.xyz[local_idx], window.transform)
                uv = window.uv[local_idx]
                ok = (
                    window.valid[local_idx]
                    & np.isfinite(xyz).all(axis=1)
                    & np.isfinite(uv).all(axis=1)
                    & (uv[:, 0] >= 0.0)
                    & (uv[:, 0] <= 1.0)
                    & (uv[:, 1] >= 0.0)
                    & (uv[:, 1] <= 1.0)
                    & (window.visibility[local_idx] >= self.min_visibility)
                    & (window.confidence[local_idx] >= self.min_confidence)
                )
                total += int(ok.shape[0])
                if not np.any(ok):
                    continue
                xy = np.stack(
                    [
                        uv[ok, 0] * float(max(w - 1, 1)),
                        uv[ok, 1] * float(max(h - 1, 1)),
                    ],
                    axis=1,
                )
                world, world_ok = backproject_xy_world(stream, int(frame_id), xy)
                if not np.any(world_ok):
                    continue
                source_parts.append(xyz[ok][world_ok])
                target_parts.append(world[world_ok])
                valid_total += int(np.count_nonzero(world_ok))
        if not source_parts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), {
                "anchor_candidates": int(total),
                "anchor_valid": 0,
            }
        source = np.concatenate(source_parts, axis=0)
        target = np.concatenate(target_parts, axis=0)
        if source.shape[0] > self.max_anchors:
            keep = np.linspace(0, source.shape[0] - 1, num=self.max_anchors, dtype=np.int64)
            source = source[keep]
            target = target[keep]
        return source, target, {"anchor_candidates": int(total), "anchor_valid": int(valid_total)}

    def _self_stitch_transforms(self, windows: list[_WindowData]) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
        if "scale_normalized" in self.mode:
            return self._self_stitch_scale_normalized_transforms(windows)
        transforms: list[dict[str, Any] | None] = [None for _ in windows]
        pair_scales: list[float] = []
        pair_p90: list[float] = []
        pair_inlier_abs010: list[float] = []
        match_stats: list[dict[str, Any]] = []
        fail_count = 0
        for idx in range(1, len(windows)):
            prev = windows[idx - 1]
            curr = windows[idx]
            match = match_overlap_carriers(
                self._window_match_payload(prev),
                self._window_match_payload(curr),
                min_visibility=self.min_visibility,
                min_confidence=self.min_confidence,
                uv_radius=self.stitch_uv_radius,
                max_matches_per_frame=self.stitch_max_matches_per_frame,
            )
            match_stats.append(match.stats)
            if match.curr_xyz.shape[0] == 0:
                fail_count += 1
                continue
            src = match.curr_xyz.reshape(-1, 3)
            dst = _apply_fit(match.prev_xyz.reshape(-1, 3), transforms[idx - 1])
            if src.shape[0] < 4:
                fail_count += 1
                continue
            try:
                fit = fit_sim3_with_diagnostics(src, dst)
                fit = self._trim_pair_fit(src, dst, fit)
            except Exception:
                fail_count += 1
                continue
            transforms[idx] = fit
            pair_scales.append(float(fit["scale"]))
            pair_p90.append(float(fit["residual_p90"]))
            pair_inlier_abs010.append(float(fit["inlier_ratio_abs010"]))
        return transforms, {
            "self_stitch_pair_count": int(max(len(windows) - 1, 0)),
            "self_stitch_fail_count": int(fail_count),
            "self_stitch_uv_radius": float(self.stitch_uv_radius),
            "self_stitch_max_matches_per_frame": int(self.stitch_max_matches_per_frame),
            "self_stitch_fit_trim_percentile": float(self.stitch_fit_trim_percentile),
            "self_stitch_scale_std": float(np.std(pair_scales)) if pair_scales else None,
            "self_stitch_residual_p90_mean": float(np.mean(pair_p90)) if pair_p90 else None,
            "self_stitch_inlier_ratio_abs010_mean": float(np.mean(pair_inlier_abs010)) if pair_inlier_abs010 else None,
            **self._aggregate_match_stats(match_stats),
        }

    def _trim_pair_fit(self, source: np.ndarray, target: np.ndarray, fit: dict[str, Any]) -> dict[str, Any]:
        trim = float(self.stitch_fit_trim_percentile)
        if not (0.0 < trim < 100.0):
            return fit
        residual = np.asarray(fit.get("residual", []), dtype=np.float64)
        if residual.size < 16:
            return fit
        keep = residual <= float(np.percentile(residual, trim))
        kept = int(np.count_nonzero(keep))
        if kept < 4 or kept >= residual.size:
            return fit
        trimmed = fit_sim3_with_diagnostics(source[keep], target[keep])
        trimmed["fit_trim_percentile"] = trim
        trimmed["fit_anchor_count"] = int(residual.size)
        trimmed["fit_kept_anchor_count"] = kept
        return trimmed

    @staticmethod
    def _window_match_payload(window: _WindowData) -> dict[str, Any]:
        return {
            "frame_ids": window.frame_ids,
            "xyz": window.xyz,
            "uv": window.uv,
            "valid": window.valid,
            "visibility": window.visibility,
            "confidence": window.confidence,
            "carrier_id": window.carrier_id,
            "persistent_tube_id": window.persistent_tube_id,
            "src_frame_global": window.src_frame_global,
            "src_xy": window.src_xy,
        }

    @staticmethod
    def _aggregate_match_stats(stats_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not stats_rows:
            return {}
        sum_keys = [
            "match_source_stable_id_count",
            "match_source_same_source_pixel_count",
            "match_source_mutual_uv_count",
            "match_source_rejected_count",
            "overlap_anchor_count",
            "mutual_uv_cycle_candidate_count",
        ]
        mean_keys = [
            "stable_id_match_ratio",
            "same_source_pixel_match_ratio",
            "mutual_uv_match_ratio",
            "cycle_consistency_pass_ratio",
        ]
        out: dict[str, Any] = {}
        for key in sum_keys:
            out[f"self_stitch_{key}"] = int(sum(int(row.get(key, 0) or 0) for row in stats_rows))
        for key in mean_keys:
            vals = [float(row[key]) for row in stats_rows if row.get(key) is not None]
            out[f"self_stitch_{key}_mean"] = float(np.mean(vals)) if vals else None
        out["self_stitch_appearance_consistency_available"] = bool(any(row.get("appearance_consistency_available") for row in stats_rows))
        vals = [float(row["appearance_consistency_pass_ratio"]) for row in stats_rows if row.get("appearance_consistency_pass_ratio") is not None]
        out["self_stitch_appearance_consistency_pass_ratio_mean"] = float(np.mean(vals)) if vals else None
        return out

    def _collect_pair_matches(self, prev: _WindowData, curr: _WindowData) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        match = match_overlap_carriers(
            self._window_match_payload(prev),
            self._window_match_payload(curr),
            min_visibility=self.min_visibility,
            min_confidence=self.min_confidence,
            uv_radius=self.stitch_uv_radius,
            max_matches_per_frame=self.stitch_max_matches_per_frame,
        )
        return match.curr_xyz.reshape(-1, 3), match.prev_xyz.reshape(-1, 3), match.stats

    def _self_stitch_scale_normalized_transforms(self, windows: list[_WindowData]) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
        transforms: list[dict[str, Any] | None] = [None for _ in windows]
        pair_payloads: list[dict[str, Any]] = []
        match_stats: list[dict[str, Any]] = []
        fail_count = 0
        for idx in range(1, len(windows)):
            src, dst, stats = self._collect_pair_matches(windows[idx - 1], windows[idx])
            match_stats.append(stats)
            if src.shape[0] < 4:
                fail_count += 1
                continue
            try:
                fit = fit_sim3_with_diagnostics(src, dst)
                fit = self._trim_pair_fit(src, dst, fit)
            except Exception:
                fail_count += 1
                continue
            pair_payloads.append({"idx": idx, "source": src, "target": dst, "fit": fit})
        scales = [float(item["fit"]["scale"]) for item in pair_payloads]
        scale_bias = float(np.exp(np.mean(np.log(np.asarray(scales, dtype=np.float64))))) if scales else None
        pair_p90: list[float] = []
        normalized_scales: list[float] = []
        for item in pair_payloads:
            idx = int(item["idx"])
            fit = item["fit"]
            if scale_bias and scale_bias > 0.0:
                pair_fit = _fit_with_fixed_scale(
                    item["source"],
                    item["target"],
                    fit,
                    float(fit["scale"]) / scale_bias,
                )
            else:
                pair_fit = fit
            transforms[idx] = _compose_fits(transforms[idx - 1], pair_fit)
            normalized_scales.append(float(pair_fit["scale"]))
            residual = np.asarray(pair_fit.get("residual", []), dtype=np.float64)
            if residual.size:
                pair_p90.append(float(np.percentile(residual, 90)))
        return transforms, {
            "self_stitch_pair_count": int(max(len(windows) - 1, 0)),
            "self_stitch_fail_count": int(fail_count),
            "self_stitch_uv_radius": float(self.stitch_uv_radius),
            "self_stitch_max_matches_per_frame": int(self.stitch_max_matches_per_frame),
            "self_stitch_fit_trim_percentile": float(self.stitch_fit_trim_percentile),
            "self_stitch_scale_normalized": True,
            "self_stitch_scale_bias_removed": scale_bias,
            "self_stitch_original_scale_std": float(np.std(scales)) if scales else None,
            "self_stitch_scale_std": float(np.std(normalized_scales)) if normalized_scales else None,
            "self_stitch_accumulated_scale_drift": float(abs(np.prod(normalized_scales) - 1.0)) if normalized_scales else None,
            "self_stitch_residual_p90_mean": float(np.mean(pair_p90)) if pair_p90 else None,
            **self._aggregate_match_stats(match_stats),
        }

    def _load_scene(self, scene: str) -> dict[str, Any]:
        if scene in self._scene_cache:
            return self._scene_cache[scene]
        scene_dir = self.debug_root / scene
        carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
        if not carrier_paths:
            raise FileNotFoundError(f"No D4RT carrier windows under {scene_dir}")
        stream = ScanNetStream(seq_name=scene, backbone="Cropformer")
        frame_ids_by_path: dict[Path, list[int]] = {}
        loaded: list[tuple[Path, dict[str, np.ndarray]]] = []
        for carrier_path in carrier_paths:
            data = self._load_npz(carrier_path)
            frame_ids_by_path[carrier_path] = self._frame_ids_for_window(
                carrier_path,
                data,
                int(np.asarray(data["uv_pred"]).shape[0]),
            )
            loaded.append((carrier_path, data))

        scene_fit = None
        anchor_diag: dict[str, Any] = {}
        if self.mode.startswith("eval_sim3"):
            source, target, anchor_diag = self._collect_eval_anchors(stream, carrier_paths, frame_ids_by_path)
            scene_fit = fit_transform(source, target, robust_trim_percentile=self.robust_trim_percentile)

        windows: list[_WindowData] = []
        for carrier_path, data in loaded:
            xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
            uv = np.asarray(data["uv_pred"], dtype=np.float32)
            valid = np.asarray(data.get("valid", np.ones(xyz.shape[:2], dtype=bool)), dtype=bool)
            visibility = np.asarray(data.get("visibility_prob", np.ones(xyz.shape[:2], dtype=np.float32)), dtype=np.float32)
            confidence = np.asarray(data.get("confidence_prob", np.ones(xyz.shape[:2], dtype=np.float32)), dtype=np.float32)
            transform = scene_fit if self.mode.startswith("eval_sim3") else None
            windows.append(
                _WindowData(
                    path=carrier_path,
                    frame_ids=frame_ids_by_path[carrier_path],
                    xyz=xyz,
                    uv=uv,
                    valid=valid,
                    visibility=visibility,
                    confidence=confidence,
                    carrier_id=np.asarray(data.get("carrier_id", np.arange(xyz.shape[1])), dtype=np.int64),
                    persistent_tube_id=np.asarray(data.get("persistent_tube_id", np.full((xyz.shape[1],), -1)), dtype=np.int64),
                    src_frame_global=np.asarray(data.get("src_frame_global", np.full((xyz.shape[1],), -1)), dtype=np.int64),
                    src_xy=np.asarray(data.get("src_xy", np.full((xyz.shape[1], 2), -1)), dtype=np.int64),
                    transform=transform,
                )
            )

        stitch_diag: dict[str, Any] = {}
        if self.mode.startswith("self_stitched"):
            transforms, stitch_diag = self._self_stitch_transforms(windows)
            for window, transform in zip(windows, transforms):
                window.transform = transform
            if "eval_sim3" in self.mode:
                source, target, anchor_diag = self._collect_eval_anchors_from_windows(stream, windows)
                scene_fit = fit_transform(source, target, robust_trim_percentile=self.robust_trim_percentile)
                for window in windows:
                    window.transform = scene_fit if window.transform is None else _compose_fits(scene_fit, window.transform)

        all_points = [_apply_fit(window.xyz.reshape(-1, 3), window.transform) for window in windows]
        spacing = _spacing_q75(np.concatenate(all_points, axis=0)) if all_points else None
        cache = {
            "scene": scene,
            "windows": windows,
            "spacing_q75": spacing,
            "scene_fit": _fit_summary(scene_fit),
            "anchor_diag": anchor_diag,
            "stitch_diag": stitch_diag,
        }
        self._scene_cache[scene] = cache
        return cache

    def _frame_window_score(self, window: _WindowData, local_idx: int) -> tuple[float, int, float]:
        xyz = window.xyz[local_idx]
        uv = window.uv[local_idx]
        ok = (
            window.valid[local_idx]
            & np.isfinite(xyz).all(axis=1)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
            & (window.visibility[local_idx] >= self.min_visibility)
            & (window.confidence[local_idx] >= self.min_confidence)
        )
        if not np.any(ok):
            return (-1.0, 0, -float("inf"))
        quality = np.asarray(window.visibility[local_idx][ok] * window.confidence[local_idx][ok], dtype=np.float32)
        center = (len(window.frame_ids) - 1) / 2.0
        return (float(np.mean(quality)), int(np.count_nonzero(ok)), -abs(float(local_idx) - center))

    def _windows_for_frame(self, windows: list[_WindowData], frame_id: int) -> list[tuple[_WindowData, int]]:
        candidates: list[tuple[_WindowData, int]] = []
        for window in windows:
            if int(frame_id) not in window.frame_ids:
                continue
            candidates.append((window, window.frame_ids.index(int(frame_id))))
        if self.overlap_policy == "all_window_union" or len(candidates) <= 1:
            return candidates
        if self.overlap_policy == "newest_window":
            return [candidates[-1]]
        if self.overlap_policy == "lowest_residual":
            scored: list[tuple[float, tuple[_WindowData, int]]] = []
            for item in candidates:
                transform = item[0].transform
                residual = None if transform is None else transform.get("residual")
                if residual is None:
                    scored.append((float("inf"), item))
                    continue
                vals = np.asarray(residual, dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                scored.append((float(np.percentile(vals, 90)) if vals.size else float("inf"), item))
            finite = [item for item in scored if np.isfinite(item[0])]
            if finite:
                return [min(finite, key=lambda item: item[0])[1]]
        best = max(candidates, key=lambda item: self._frame_window_score(item[0], item[1]))
        return [best]

    def project_frame_masks(
        self,
        *,
        dataset: object,
        scene_points: np.ndarray,
        mask_image: np.ndarray,
        frame_id: int,
        depth_max_pre: float,
    ) -> FrameProjection:
        del depth_max_pre
        scene = str(getattr(dataset, "seq_name", ""))
        cache = self._load_scene(scene)
        scene_points_np = _as_numpy_points(scene_points)
        scene_tree = cKDTree(scene_points_np)
        mask_np = np.asarray(mask_image)
        height, width = mask_np.shape[:2]
        radius = self.nn_radius
        if self.mode.endswith("_density") and cache.get("spacing_q75") is not None:
            radius = max(radius, float(cache["spacing_q75"]) * self.density_alpha)

        mask_info: dict[int, set[int]] = defaultdict(set)
        frame_point_ids: set[int] = set()
        local_points = 0
        positive_points = 0
        hit_points = 0
        interior_filtered_points = 0
        candidate_windows = [
            (window, window.frame_ids.index(int(frame_id)))
            for window in cache["windows"]
            if int(frame_id) in window.frame_ids
        ]
        selected_windows = self._windows_for_frame(cache["windows"], int(frame_id))
        source_windows = 0
        for window, local_idx in selected_windows:
            source_windows += 1
            xyz = _apply_fit(window.xyz[local_idx], window.transform)
            uv = window.uv[local_idx]
            ok = (
                window.valid[local_idx]
                & np.isfinite(xyz).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (window.visibility[local_idx] >= self.min_visibility)
                & (window.confidence[local_idx] >= self.min_confidence)
            )
            local_points += int(np.count_nonzero(ok))
            if not np.any(ok):
                continue
            x = np.rint(uv[ok, 0] * float(max(width - 1, 1))).astype(np.int64)
            y = np.rint(uv[ok, 1] * float(max(height - 1, 1))).astype(np.int64)
            in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            if not np.any(in_bounds):
                continue
            points = xyz[ok][in_bounds]
            mask_ids = mask_np[y[in_bounds], x[in_bounds]].astype(np.int64)
            positive = mask_ids > 0
            positive_points += int(np.count_nonzero(positive))
            if not np.any(positive):
                continue
            positive_idx = np.flatnonzero(positive)
            if self.min_mask_interior_px > 0.0:
                interior_dist = _mask_interior_distances(
                    mask_np,
                    x[in_bounds][positive],
                    y[in_bounds][positive],
                    mask_ids[positive],
                )
                keep_interior = interior_dist >= self.min_mask_interior_px
                interior_filtered_points += int(np.count_nonzero(~keep_interior))
                if not np.any(keep_interior):
                    continue
                positive_idx = positive_idx[keep_interior]
            dist, nn_idx = scene_tree.query(points[positive_idx], k=self.nn_k, distance_upper_bound=radius)
            dist = np.asarray(dist)
            nn_idx = np.asarray(nn_idx)
            if self.nn_k == 1:
                dist = dist[:, None]
                nn_idx = nn_idx[:, None]
            hit = np.isfinite(dist) & (nn_idx < scene_points_np.shape[0])
            if self.local_outlier_filter and np.any(hit):
                hit_any = np.any(hit, axis=1)
                nearest_hit_dist = np.min(np.where(hit, dist, np.inf), axis=1)
                dist_hit = nearest_hit_dist[hit_any]
                cutoff = min(float(radius), float(np.percentile(dist_hit, 90)))
                hit &= dist <= cutoff
            hit_any = np.any(hit, axis=1)
            hit_points += int(np.count_nonzero(hit_any))
            for row_idx, mask_id in enumerate(mask_ids[positive_idx].tolist()):
                cols = nn_idx[row_idx][hit[row_idx]]
                if cols.size == 0:
                    continue
                mid = int(mask_id)
                for point_id in cols.tolist():
                    pid = int(point_id)
                    mask_info[mid].add(pid)
                    frame_point_ids.add(pid)

        diag = {
            "provider": self.name,
            "scene": scene,
            "frame_id": int(frame_id),
            "source_windows": int(source_windows),
            "selected_source_windows": int(source_windows),
            "candidate_source_windows": int(len(candidate_windows)),
            "duplicate_window_hit_rate": float(
                max(len(candidate_windows) - source_windows, 0) / max(len(candidate_windows), 1)
            ),
            "overlap_policy": self.overlap_policy,
            "local_point_count": int(local_points),
            "positive_mask_point_count": int(positive_points),
            "hit_point_count": int(hit_points),
            "interior_filtered_point_count": int(interior_filtered_points),
            "projection_hit_rate": float(hit_points / max(positive_points, 1)),
            "mask_projection_empty_rate": float(1.0 - len(mask_info) / max(len(np.unique(mask_np[mask_np > 0])), 1)),
            "mean_points_per_2d_mask": float(hit_points / max(len(mask_info), 1)),
            "nn_radius": float(radius),
            "nn_k": int(self.nn_k),
            "assigned_vertex_count": int(sum(len(v) for v in mask_info.values())),
            "unique_assigned_vertex_count": int(len(frame_point_ids)),
            "min_mask_interior_px": float(self.min_mask_interior_px),
            **cache.get("scene_fit", {}),
            **cache.get("anchor_diag", {}),
            **cache.get("stitch_diag", {}),
        }
        self.frame_diagnostics.append(diag)
        return FrameProjection(
            mask_info={int(k): set(v) for k, v in mask_info.items()},
            frame_point_ids=sorted(frame_point_ids),
            depth_max=0.0,
            diagnostics=diag,
        )
