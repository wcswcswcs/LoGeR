from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.rescore_scannet import verify_object_dict_prediction_alignment
from stream4d.scannet_stream import ScanNetStream


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
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


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


class SilhouetteScorer:
    def __init__(self, stream: ScanNetStream, scene_points: np.ndarray, args: argparse.Namespace) -> None:
        self.stream = stream
        self.scene_points = np.asarray(scene_points, dtype=np.float32)
        self.args = args
        self.intrinsics = stream.load_intrinsics()
        self._depth_cache: dict[int, np.ndarray] = {}
        self._mask_cache: dict[int, np.ndarray] = {}
        self._pose_inv_cache: dict[int, np.ndarray | None] = {}
        self._distance_cache: dict[tuple[int, int], np.ndarray] = {}

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

    def _distance_inside_mask(self, frame_id: int, mask_id: int) -> np.ndarray:
        key = (int(frame_id), int(mask_id))
        if key not in self._distance_cache:
            mask_bool = self._mask(frame_id) == int(mask_id)
            self._distance_cache[key] = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
        return self._distance_cache[key]

    def _project(self, frame_id: int, point_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if point_ids.size == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        pose_inv = self._pose_inv(frame_id)
        if pose_inv is None:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        depth = self._depth(frame_id)
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

    @staticmethod
    def _unique_observations(mask_list: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        best: dict[tuple[int, int], float] = {}
        for frame_id, mask_id, coverage in mask_list:
            key = (int(frame_id), int(mask_id))
            best[key] = max(best.get(key, 0.0), float(coverage))
        return sorted(
            [(frame_id, mask_id, coverage) for (frame_id, mask_id), coverage in best.items()],
            key=lambda item: float(item[2]),
            reverse=True,
        )

    def score_object(self, value: dict) -> dict[str, float]:
        point_ids = np.asarray(value.get("point_ids", []), dtype=np.int64).reshape(-1)
        if point_ids.size == 0:
            return {
                "silhouette_quality": 0.0,
                "inside_visible_ratio": 0.0,
                "inside_projected_ratio": 0.0,
                "interior_ratio": 0.0,
                "visible_points": 0.0,
                "projected_points": 0.0,
                "used_observations": 0.0,
            }
        if int(self.args.max_points_per_object) > 0 and point_ids.shape[0] > int(self.args.max_points_per_object):
            keep = np.linspace(0, point_ids.shape[0] - 1, int(self.args.max_points_per_object), dtype=np.int64)
            point_ids = point_ids[keep]
        observations = self._unique_observations(list(value.get("mask_list", [])))
        if int(self.args.max_observations) > 0:
            observations = observations[: int(self.args.max_observations)]

        visible_total = 0
        projected_total = 0
        inside_visible_total = 0
        inside_projected_total = 0
        interior_total = 0
        used_observations = 0
        for frame_id, mask_id, _ in observations:
            xy, z, _ = self._project(int(frame_id), point_ids)
            if xy.size == 0:
                continue
            depth = self._depth(int(frame_id))
            mask = self._mask(int(frame_id))
            obs_depth = depth[xy[:, 1], xy[:, 0]]
            projected = np.isfinite(obs_depth) & (obs_depth > 0.0)
            if not np.any(projected):
                continue
            projected_total += int(np.count_nonzero(projected))
            depth_error = np.abs(obs_depth - z)
            visible = projected & (depth_error <= float(self.args.depth_tolerance))
            if int(np.count_nonzero(visible)) < int(self.args.min_visible_points):
                continue
            used_observations += 1
            visible_total += int(np.count_nonzero(visible))
            inside = mask[xy[:, 1], xy[:, 0]] == int(mask_id)
            inside_projected_total += int(np.count_nonzero(projected & inside))
            inside_visible = visible & inside
            inside_visible_total += int(np.count_nonzero(inside_visible))
            if np.any(inside_visible):
                distance = self._distance_inside_mask(int(frame_id), int(mask_id))
                margin = distance[xy[inside_visible, 1], xy[inside_visible, 0]]
                interior_total += int(np.count_nonzero(margin >= float(self.args.boundary_margin_px)))

        inside_visible_ratio = float(inside_visible_total / max(visible_total, 1))
        inside_projected_ratio = float(inside_projected_total / max(projected_total, 1))
        interior_ratio = float(interior_total / max(inside_visible_total, 1))
        observation_weight = min(1.0, float(used_observations) / max(float(self.args.observation_saturation), 1.0))
        visible_weight = min(1.0, float(visible_total) / max(float(self.args.visible_saturation), 1.0))
        silhouette_quality = (
            inside_visible_ratio
            * (0.5 + 0.5 * interior_ratio)
            * (0.5 + 0.5 * observation_weight)
            * (0.5 + 0.5 * visible_weight)
        )
        return {
            "silhouette_quality": float(silhouette_quality),
            "inside_visible_ratio": inside_visible_ratio,
            "inside_projected_ratio": inside_projected_ratio,
            "interior_ratio": interior_ratio,
            "visible_points": float(visible_total),
            "projected_points": float(projected_total),
            "used_observations": float(used_observations),
        }


def _compose_scores(
    source_scores: np.ndarray,
    point_areas: np.ndarray,
    silhouette_quality: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    score_norm = _normalize(source_scores)
    area_norm = _normalize(np.log1p(np.maximum(point_areas, 0.0)))
    if args.quality_mode == "silhouette":
        scores = silhouette_quality
    elif args.quality_mode == "score_silhouette":
        scores = float(args.score_weight) * score_norm + (1.0 - float(args.score_weight)) * silhouette_quality
    elif args.quality_mode == "silhouette_area":
        scores = float(args.silhouette_weight) * silhouette_quality + (1.0 - float(args.silhouette_weight)) * area_norm
    elif args.quality_mode == "score_silhouette_area":
        remain = max(0.0, 1.0 - float(args.score_weight) - float(args.silhouette_weight))
        scores = float(args.score_weight) * score_norm + float(args.silhouette_weight) * silhouette_quality + remain * area_norm
    else:
        raise ValueError(f"Unsupported quality mode: {args.quality_mode}")
    return scores.astype(np.float32)


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    root = Path(args.root)
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    object_path = stream.object_dir / args.input_config / "object_dict.npy"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not object_path.exists():
        raise FileNotFoundError(object_path)
    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        source_scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    object_dict = np.load(object_path, allow_pickle=True).item()
    object_items = [(int(k), v) for k, v in sorted(object_dict.items(), key=lambda item: int(item[0]))]
    if len(object_items) != masks.shape[1]:
        raise RuntimeError(f"{seq_name}: object_dict count {len(object_items)} != prediction columns {masks.shape[1]}")
    alignment = verify_object_dict_prediction_alignment(masks, object_items, threshold=0.99)
    if alignment["alignment_checked"] and int(alignment["alignment_failed_instances"]) > 0:
        raise RuntimeError(f"{seq_name}: object_dict/prediction alignment failed: {alignment}")

    # Open3D is intentionally imported lazily to keep --help and parser startup light.
    import open3d as o3d

    scene_points = np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)
    scorer = SilhouetteScorer(stream, scene_points, args)
    records = [scorer.score_object(value) for _, value in object_items]
    silhouette_quality = np.asarray([item["silhouette_quality"] for item in records], dtype=np.float32)
    point_areas = masks.sum(axis=0).astype(np.float32)
    scores = _compose_scores(source_scores, point_areas, silhouette_quality, args)

    keep = np.ones((masks.shape[1],), dtype=bool)
    if float(args.min_silhouette_quality) > 0.0:
        keep &= silhouette_quality >= float(args.min_silhouette_quality)
    if int(args.min_visible_points_total) > 0:
        visible = np.asarray([item["visible_points"] for item in records], dtype=np.float32)
        keep &= visible >= float(args.min_visible_points_total)
    if int(args.max_instances) > 0 and int(np.count_nonzero(keep)) > int(args.max_instances):
        keep_indices = np.flatnonzero(keep)
        order = keep_indices[np.argsort(-scores[keep_indices], kind="stable")[: int(args.max_instances)]]
        new_keep = np.zeros_like(keep)
        new_keep[order] = True
        keep = new_keep

    masks_out = masks[:, keep]
    scores_out = scores[keep]
    classes_out = classes[keep]
    kept_items = [object_items[idx] for idx in np.flatnonzero(keep).tolist()]
    order = np.argsort(-scores_out, kind="stable")
    masks_out = masks_out[:, order]
    scores_out = scores_out[order]
    classes_out = classes_out[order]
    kept_items = [kept_items[idx] for idx in order.tolist()]

    pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{seq_name}.npz",
        pred_masks=masks_out,
        pred_score=scores_out,
        pred_classes=classes_out,
    )

    object_out = stream.object_dir / args.output_config
    object_out.mkdir(parents=True, exist_ok=True)
    np.save(object_out / "object_dict.npy", {object_id: value for object_id, value in kept_items}, allow_pickle=True)

    tmp_in = _tmp_path(root, args.input_config, seq_name)
    tmp_out = _tmp_path(root, args.output_config, seq_name)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
    else:
        np.save(tmp_out, np.flatnonzero(masks_out.any(axis=1)).astype(np.int64))

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "num_instances_before": int(masks.shape[1]),
        "num_instances_after": int(masks_out.shape[1]),
        "num_removed": int(masks.shape[1] - masks_out.shape[1]),
        "silhouette_quality_min": float(np.min(silhouette_quality)) if silhouette_quality.size else 0.0,
        "silhouette_quality_mean": float(np.mean(silhouette_quality)) if silhouette_quality.size else 0.0,
        "silhouette_quality_max": float(np.max(silhouette_quality)) if silhouette_quality.size else 0.0,
        "inside_visible_ratio_mean": float(np.mean([item["inside_visible_ratio"] for item in records])) if records else 0.0,
        "inside_projected_ratio_mean": float(np.mean([item["inside_projected_ratio"] for item in records])) if records else 0.0,
        "interior_ratio_mean": float(np.mean([item["interior_ratio"] for item in records])) if records else 0.0,
        "visible_points_mean": float(np.mean([item["visible_points"] for item in records])) if records else 0.0,
        "used_observations_mean": float(np.mean([item["used_observations"] for item in records])) if records else 0.0,
        "score_min": float(np.min(scores_out)) if scores_out.size else 0.0,
        "score_mean": float(np.mean(scores_out)) if scores_out.size else 0.0,
        "score_max": float(np.max(scores_out)) if scores_out.size else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score object predictions by multi-view silhouette consistency.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--quality-mode", default="score_silhouette", choices=[
        "silhouette",
        "score_silhouette",
        "silhouette_area",
        "score_silhouette_area",
    ])
    parser.add_argument("--score-weight", type=float, default=0.75)
    parser.add_argument("--silhouette-weight", type=float, default=0.50)
    parser.add_argument("--max-observations", type=int, default=8)
    parser.add_argument("--max-points-per-object", type=int, default=2000)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--boundary-margin-px", type=float, default=2.0)
    parser.add_argument("--min-visible-points", type=int, default=3)
    parser.add_argument("--visible-saturation", type=float, default=200.0)
    parser.add_argument("--observation-saturation", type=float, default=5.0)
    parser.add_argument("--min-silhouette-quality", type=float, default=0.0)
    parser.add_argument("--min-visible-points-total", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--summary-root", default="outputs/silhouette_consistency_score")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    aggregate: dict[str, float] = {}
    if rows:
        numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
        for key in numeric_keys:
            aggregate[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    summary = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.output_config}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
