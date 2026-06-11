from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    dirname = config if config.endswith(suffix) else f"{config}{suffix}"
    return root / "data" / "prediction" / dirname / f"{seq_name}.npz"


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


class ProjectionCache:
    def __init__(self, stream: ScanNetStream, scene_points: np.ndarray) -> None:
        self.stream = stream
        self.scene_points = np.asarray(scene_points, dtype=np.float32)
        self.intrinsics = stream.load_intrinsics().astype(np.float32)
        self._depth_cache: dict[int, np.ndarray] = {}
        self._mask_cache: dict[int, np.ndarray] = {}
        self._pose_inv_cache: dict[int, np.ndarray | None] = {}
        self._distance_cache: dict[tuple[int, int], np.ndarray] = {}

    def depth(self, frame_id: int) -> np.ndarray:
        frame_id = int(frame_id)
        if frame_id not in self._depth_cache:
            self._depth_cache[frame_id] = self.stream.load_depth(frame_id)
        return self._depth_cache[frame_id]

    def mask(self, frame_id: int) -> np.ndarray:
        frame_id = int(frame_id)
        if frame_id not in self._mask_cache:
            depth = self.depth(frame_id)
            mask = self.stream.load_mask(frame_id)
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._mask_cache[frame_id] = mask
        return self._mask_cache[frame_id]

    def pose_inv(self, frame_id: int) -> np.ndarray | None:
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

    def distance_inside_mask(self, frame_id: int, mask_id: int) -> np.ndarray:
        key = (int(frame_id), int(mask_id))
        if key not in self._distance_cache:
            mask_bool = self.mask(frame_id) == int(mask_id)
            self._distance_cache[key] = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
        return self._distance_cache[key]

    def project(self, frame_id: int, point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_ids = point_ids.astype(np.int64, copy=False)
        if point_ids.size == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        pose_inv = self.pose_inv(frame_id)
        if pose_inv is None:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        depth = self.depth(frame_id)
        h, w = depth.shape
        point_ids = point_ids[(point_ids >= 0) & (point_ids < self.scene_points.shape[0])]
        if point_ids.size == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        pts = self.scene_points[point_ids]
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        cam = (pose_inv @ pts_h.T).T[:, :3]
        z = cam[:, 2]
        valid = np.isfinite(cam).all(axis=1) & (z > 1e-6)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        valid_ids = point_ids[valid]
        cam = cam[valid]
        z = z[valid]
        fx = float(self.intrinsics[0, 0])
        fy = float(self.intrinsics[1, 1])
        cx = float(self.intrinsics[0, 2])
        cy = float(self.intrinsics[1, 2])
        xs = np.rint((cam[:, 0] * fx / z) + cx).astype(np.int64)
        ys = np.rint((cam[:, 1] * fy / z) + cy).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(in_bounds):
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        xy = np.stack([xs[in_bounds], ys[in_bounds]], axis=1).astype(np.int64)
        return xy, z[in_bounds].astype(np.float32), valid_ids[in_bounds].astype(np.int64)


def _sample_ids(point_ids: np.ndarray, max_points: int) -> np.ndarray:
    if int(max_points) <= 0 or point_ids.shape[0] <= int(max_points):
        return point_ids
    keep = np.linspace(0, point_ids.shape[0] - 1, int(max_points), dtype=np.int64)
    return point_ids[keep]


def _discover_observation(
    cache: ProjectionCache,
    frame_id: int,
    point_ids: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, dict[str, float]] | None:
    xy, z, _ = cache.project(frame_id, point_ids)
    if xy.size == 0:
        return None
    depth = cache.depth(frame_id)
    mask = cache.mask(frame_id)
    obs_depth = depth[xy[:, 1], xy[:, 0]]
    projected = np.isfinite(obs_depth) & (obs_depth > 0.0)
    visible = projected & (np.abs(obs_depth - z) <= float(args.depth_tolerance))
    if int(np.count_nonzero(visible)) < int(args.min_visible_points):
        return None
    labels, counts = np.unique(mask[xy[visible, 1], xy[visible, 0]], return_counts=True)
    nonzero = labels > 0
    labels = labels[nonzero]
    counts = counts[nonzero]
    if labels.size == 0:
        return None
    best_pos = int(np.argmax(counts))
    best_label = int(labels[best_pos])
    best_count = int(counts[best_pos])
    dominant_ratio = float(best_count / max(int(np.count_nonzero(visible)), 1))
    if best_count < int(args.min_dominant_points) or dominant_ratio < float(args.min_dominant_ratio):
        return None
    return best_label, {
        "visible": float(np.count_nonzero(visible)),
        "dominant_count": float(best_count),
        "dominant_ratio": dominant_ratio,
    }


def _refine_instance(
    cache: ProjectionCache,
    frame_ids: list[int],
    point_ids: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    if point_ids.size == 0:
        return point_ids, {
            "used_observations": 0.0,
            "visible_points_mean": 0.0,
            "inside_ratio_mean": 0.0,
            "interior_ratio_mean": 0.0,
        }
    sample_ids = _sample_ids(point_ids, int(args.discovery_max_points))
    visible_counts = np.zeros((point_ids.shape[0],), dtype=np.int16)
    inside_counts = np.zeros((point_ids.shape[0],), dtype=np.int16)
    interior_counts = np.zeros((point_ids.shape[0],), dtype=np.int16)
    id_to_local = {int(pid): idx for idx, pid in enumerate(point_ids.tolist())}
    used_observations = 0
    visible_per_obs: list[int] = []
    inside_ratios: list[float] = []
    interior_ratios: list[float] = []

    for frame_id in frame_ids:
        obs = _discover_observation(cache, frame_id, sample_ids, args)
        if obs is None:
            continue
        mask_id, _ = obs
        xy, z, projected_ids = cache.project(frame_id, point_ids)
        if xy.size == 0:
            continue
        depth = cache.depth(frame_id)
        mask = cache.mask(frame_id)
        obs_depth = depth[xy[:, 1], xy[:, 0]]
        projected = np.isfinite(obs_depth) & (obs_depth > 0.0)
        visible = projected & (np.abs(obs_depth - z) <= float(args.depth_tolerance))
        if int(np.count_nonzero(visible)) < int(args.min_visible_points):
            continue
        local_idx = np.asarray([id_to_local[int(pid)] for pid in projected_ids], dtype=np.int64)
        inside = mask[xy[:, 1], xy[:, 0]] == int(mask_id)
        visible_local = local_idx[visible]
        inside_visible = visible & inside
        inside_local = local_idx[inside_visible]
        visible_counts[visible_local] += 1
        inside_counts[inside_local] += 1
        if np.any(inside_visible):
            distance = cache.distance_inside_mask(frame_id, mask_id)
            margins = distance[xy[inside_visible, 1], xy[inside_visible, 0]]
            interior = margins >= float(args.boundary_margin_px)
            interior_counts[inside_local[interior]] += 1
        used_observations += 1
        visible_num = int(np.count_nonzero(visible))
        inside_num = int(np.count_nonzero(inside_visible))
        interior_num = int(np.count_nonzero(interior_counts[inside_local] > 0)) if inside_local.size else 0
        visible_per_obs.append(visible_num)
        inside_ratios.append(float(inside_num / max(visible_num, 1)))
        interior_ratios.append(float(interior_num / max(inside_num, 1)))
        if int(args.max_observations) > 0 and used_observations >= int(args.max_observations):
            break

    enough_visible = visible_counts >= int(args.min_point_visible_views)
    inside_ratio = inside_counts.astype(np.float32) / np.maximum(visible_counts.astype(np.float32), 1.0)
    interior_ratio = interior_counts.astype(np.float32) / np.maximum(visible_counts.astype(np.float32), 1.0)
    supported = enough_visible & (inside_ratio >= float(args.min_point_inside_ratio))
    if float(args.min_point_interior_ratio) > 0.0:
        supported &= interior_ratio >= float(args.min_point_interior_ratio)
    if args.unobserved_policy == "keep":
        keep = (~enough_visible) | supported
    else:
        keep = supported
    if int(np.count_nonzero(keep)) < int(args.min_points_after_refine):
        keep = np.ones_like(keep, dtype=bool)

    return point_ids[keep], {
        "used_observations": float(used_observations),
        "visible_points_mean": float(np.mean(visible_per_obs)) if visible_per_obs else 0.0,
        "inside_ratio_mean": float(np.mean(inside_ratios)) if inside_ratios else 0.0,
        "interior_ratio_mean": float(np.mean(interior_ratios)) if interior_ratios else 0.0,
        "point_keep_ratio": float(np.count_nonzero(keep) / max(point_ids.shape[0], 1)),
        "points_before": float(point_ids.shape[0]),
        "points_after": float(np.count_nonzero(keep)),
    }


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    root = Path(args.root)
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D")
    if scores.shape[0] != masks.shape[1] or classes.shape[0] != masks.shape[1]:
        raise ValueError(f"{seq_name}: inconsistent prediction arrays")

    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone, root=root / args.scannet_root)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    scene_points = np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)
    if scene_points.shape[0] != masks.shape[0]:
        raise RuntimeError(f"{seq_name}: mesh points {scene_points.shape[0]} != prediction rows {masks.shape[0]}")
    cache = ProjectionCache(stream, scene_points)
    frame_ids = stream.frame_ids(stride=int(args.frame_stride), max_frames=int(args.max_frames))

    support_ids: np.ndarray | None = None
    if args.refine_support_config:
        support_path = _tmp_path(root, args.refine_support_config, seq_name)
        if not support_path.exists():
            raise FileNotFoundError(support_path)
        support_ids = np.load(support_path).astype(np.int64)
        support_ids = support_ids[(support_ids >= 0) & (support_ids < masks.shape[0])]

    refined = masks.copy() if args.outside_refine_support == "keep" else np.zeros_like(masks, dtype=bool)
    records: list[dict[str, float]] = []
    changed = 0
    for idx in range(masks.shape[1]):
        point_ids = np.flatnonzero(masks[:, idx]).astype(np.int64)
        if support_ids is not None:
            point_ids = np.intersect1d(point_ids, support_ids, assume_unique=False)
        if point_ids.shape[0] < int(args.min_points_before_refine):
            if support_ids is not None and args.outside_refine_support == "drop":
                refined[support_ids, idx] = masks[support_ids, idx]
            records.append({
                "used_observations": 0.0,
                "visible_points_mean": 0.0,
                "inside_ratio_mean": 0.0,
                "interior_ratio_mean": 0.0,
                "point_keep_ratio": 1.0,
                "points_before": float(point_ids.shape[0]),
                "points_after": float(point_ids.shape[0]),
            })
            continue
        kept_ids, diag = _refine_instance(cache, frame_ids, point_ids, args)
        if support_ids is not None:
            refined[support_ids, idx] = False
        else:
            refined[:, idx] = False
        refined[kept_ids, idx] = True
        if kept_ids.shape[0] != point_ids.shape[0]:
            changed += 1
        records.append(diag)

    keep_instance = np.ones((refined.shape[1],), dtype=bool)
    if args.drop_empty:
        keep_instance &= refined.any(axis=0)
    refined = refined[:, keep_instance]
    scores_out = scores[keep_instance]
    classes_out = classes[keep_instance]

    pred_out = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out / f"{seq_name}.npz",
        pred_masks=refined,
        pred_score=scores_out,
        pred_classes=classes_out,
    )
    tmp_out = _tmp_path(root, args.output_config, seq_name)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if args.tmp_policy == "input":
        input_tmp = _tmp_path(root, args.input_config, seq_name)
        if input_tmp.exists():
            shutil.copy2(input_tmp, tmp_out)
        else:
            np.save(tmp_out, np.flatnonzero(refined.any(axis=1)).astype(np.int64))
    elif args.tmp_policy == "refine_support" and args.refine_support_config:
        shutil.copy2(_tmp_path(root, args.refine_support_config, seq_name), tmp_out)
    else:
        np.save(tmp_out, np.flatnonzero(refined.any(axis=1)).astype(np.int64))

    def mean(key: str) -> float:
        vals = [float(item[key]) for item in records if key in item]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "num_instances_before": int(masks.shape[1]),
        "num_instances_after": int(refined.shape[1]),
        "num_changed_instances": int(changed),
        "num_frames": int(len(frame_ids)),
        "union_before": int(np.count_nonzero(masks.any(axis=1))),
        "union_after": int(np.count_nonzero(refined.any(axis=1))),
        "used_observations_mean": mean("used_observations"),
        "visible_points_mean": mean("visible_points_mean"),
        "inside_ratio_mean": mean("inside_ratio_mean"),
        "interior_ratio_mean": mean("interior_ratio_mean"),
        "point_keep_ratio_mean": mean("point_keep_ratio"),
        "points_before_mean": mean("points_before"),
        "points_after_mean": mean("points_after"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary-refine predictions by self-discovering 2D mask observations.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--refine-support-config", default="")
    parser.add_argument("--outside-refine-support", default="keep", choices=["keep", "drop"])
    parser.add_argument("--frame-stride", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--max-observations", type=int, default=8)
    parser.add_argument("--discovery-max-points", type=int, default=1500)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--boundary-margin-px", type=float, default=2.0)
    parser.add_argument("--min-visible-points", type=int, default=8)
    parser.add_argument("--min-dominant-points", type=int, default=5)
    parser.add_argument("--min-dominant-ratio", type=float, default=0.35)
    parser.add_argument("--min-point-visible-views", type=int, default=1)
    parser.add_argument("--min-point-inside-ratio", type=float, default=0.50)
    parser.add_argument("--min-point-interior-ratio", type=float, default=0.0)
    parser.add_argument("--unobserved-policy", default="keep", choices=["keep", "drop"])
    parser.add_argument("--min-points-before-refine", type=int, default=20)
    parser.add_argument("--min-points-after-refine", type=int, default=10)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--tmp-policy", default="input", choices=["input", "refine_support", "recompute"])
    parser.add_argument("--eval-policy", default="own_recompute_self_discovered_boundary_refine")
    parser.add_argument("--summary-root", default="outputs/self_discovered_boundary_refine_v4_1")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    numeric_keys = sorted(
        key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.generic))
    )
    aggregate = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.generic))]
        if vals:
            aggregate[f"mean_{key}"] = float(np.mean(vals))
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    source_configs = [args.input_config]
    if args.refine_support_config:
        source_configs.append(args.refine_support_config)
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=source_configs,
        pre_points_policy=args.tmp_policy,
        support_policy=(
            "self_discovered_boundary_refine:"
            f"frame_stride={args.frame_stride}:max_obs={args.max_observations}:"
            f"inside={args.min_point_inside_ratio}:interior={args.min_point_interior_ratio}:"
            f"unobserved={args.unobserved_policy}"
        ),
        notes=(
            "Self-discovered mask-observation boundary refinement from predicted masks, "
            "RGB-D, poses, and non-GT 2D masks only."
        ),
        extra={
            "eval_policy": args.eval_policy,
            "summary_path": str(out_path),
            "input_config": args.input_config,
            "refine_support_config": args.refine_support_config,
        },
    )
    write_prediction_manifest(
        args.output_config,
        manifest,
        root=root,
        pred_suffix=args.pred_suffix.lstrip("_"),
    )
    print(f"[self-discovered-boundary-refine] wrote {out_path}")


if __name__ == "__main__":
    main()
