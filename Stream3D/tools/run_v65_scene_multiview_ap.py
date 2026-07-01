from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.v65_visualization_export import _load_scene_mesh, _prediction_owner_ids
from tools.run_v65_soma_pipeline_visualization import (
    _best_objectlet_variant,
    _load_pipeline_support,
    _project,
    _rel,
    _resolve_pipeline_mask_dir,
    _sha256,
)


AP_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.5, 0.95, 0.05)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_sha256sums(path: Path, files: list[Path]) -> None:
    lines: list[str] = []
    for file_path in sorted(set(files)):
        if not file_path.exists():
            continue
        try:
            rel_path = file_path.relative_to(ROOT)
        except ValueError:
            rel_path = file_path
        lines.append(f"{_sha256(file_path)}  {rel_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _resize_nearest(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


def _read_label_png(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label png: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return _resize_nearest(np.asarray(image, dtype=np.int64), shape_hw).astype(np.int64, copy=False)


def _load_gt_2d(scene: str, frame_id: int, shape_hw: tuple[int, int]) -> np.ndarray:
    path = ROOT / "data" / "scannet" / "processed" / scene / "instance" / "instance" / f"{int(frame_id)}.png"
    return _read_label_png(path, shape_hw)


def _soma_pred_2d(
    *,
    mask_dir: Path,
    frame_id: int,
    shape_hw: tuple[int, int],
    mask_to_object_idx: dict[tuple[int, int], int],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = mask_dir / f"{int(frame_id)}.png"
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.int64), {
            "mask_exists": False,
            "mask_path": _rel(path),
            "positive_mask_pixels": 0,
            "mapped_pred_pixels": 0,
            "mapped_mask_ids": 0,
        }
    mask = _read_label_png(path, shape_hw)
    pred = np.zeros(mask.shape, dtype=np.int64)
    ids = [int(value) for value in np.unique(mask) if int(value) > 0]
    mapped_ids = 0
    for mask_id in ids:
        object_idx = int(mask_to_object_idx.get((int(frame_id), int(mask_id)), 0))
        if object_idx <= 0:
            continue
        pred[mask == mask_id] = object_idx
        mapped_ids += 1
    return pred, {
        "mask_exists": True,
        "mask_path": _rel(path),
        "positive_mask_pixels": int(np.count_nonzero(mask > 0)),
        "mapped_pred_pixels": int(np.count_nonzero(pred > 0)),
        "mapped_mask_ids": int(mapped_ids),
    }


def _load_stream3d_vertex_labels(
    *,
    scene: str,
    config: str,
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred_path = ROOT / "data" / "prediction" / f"{config}_class_agnostic" / f"{scene}.npz"
    pre_points_path = ROOT / "data" / "TMP" / config / f"{scene}_pre_points.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing Stream3D prediction npz: {pred_path}")
    with np.load(pred_path) as payload:
        masks = np.asarray(payload["pred_masks"], dtype=bool)
        if "pred_score" in payload.files:
            scores = np.asarray(payload["pred_score"], dtype=np.float32)
            score_key = "pred_score"
        elif "pred_scores" in payload.files:
            scores = np.asarray(payload["pred_scores"], dtype=np.float32)
            score_key = "pred_scores"
        else:
            scores = np.ones((masks.shape[1],), dtype=np.float32)
            score_key = "missing_scores_constant_one_fallback"
    labels = np.zeros((vertex_count,), dtype=np.int64)
    if masks.shape[0] == vertex_count:
        owner = _prediction_owner_ids(masks, scores)
        valid_vertex_ids = np.arange(vertex_count, dtype=np.int64)
        mask_contract = "full_scene_vertex_mask"
    elif pre_points_path.exists():
        pre_points = np.asarray(np.load(pre_points_path), dtype=np.int64)
        if masks.shape[0] != pre_points.shape[0]:
            raise ValueError(f"Stream3D mask/pre_points length mismatch: masks={masks.shape[0]} pre_points={pre_points.shape[0]}")
        valid = (pre_points >= 0) & (pre_points < vertex_count)
        owner = _prediction_owner_ids(masks[valid], scores)
        valid_vertex_ids = pre_points[valid]
        mask_contract = "pre_points_vertex_mask"
    else:
        raise ValueError(f"unsupported Stream3D mask rows={masks.shape[0]} vertex_count={vertex_count}; no pre_points file")
    covered = owner >= 0
    labels[valid_vertex_ids[covered]] = owner[covered].astype(np.int64) + 1
    diag = {
        "prediction_npz": _rel(pred_path),
        "prediction_npz_sha256": _sha256(pred_path),
        "pre_points_npy": _rel(pre_points_path) if pre_points_path.exists() else "",
        "pre_points_npy_sha256": _sha256(pre_points_path) if pre_points_path.exists() else "",
        "pred_mask_shape": list(masks.shape),
        "pred_object_count": int(masks.shape[1]),
        "pred_vertex_count": int(np.count_nonzero(labels > 0)),
        "mask_contract": mask_contract,
        "score_key": score_key,
        "score_min": float(np.min(scores)) if scores.size else 0.0,
        "score_max": float(np.max(scores)) if scores.size else 0.0,
        "score_unique_count": int(np.unique(scores).shape[0]) if scores.size else 0,
    }
    return labels, scores.astype(np.float32, copy=False), diag


def _camera_points_from_depth(depth: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float32, copy=False)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    pixels = np.stack([ys, xs], axis=1).astype(np.int32, copy=False)
    return points, pixels


def _world_points_from_depth(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_points, pixels = _camera_points_from_depth(depth, intrinsics)
    if camera_points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), pixels
    rot = pose[:3, :3].astype(np.float32, copy=False)
    trans = pose[:3, 3].astype(np.float32, copy=False)
    world = camera_points @ rot.T + trans[None, :]
    return world.astype(np.float32, copy=False), pixels


def _vertex_map_cache_path(cache_root: Path, scene: str, frame_id: int, radius: float) -> Path:
    radius_tag = str(float(radius)).replace(".", "p")
    return cache_root / scene / f"nn_radius_{radius_tag}" / f"{int(frame_id)}.npz"


def _load_or_build_vertex_map(
    *,
    stream: ScanNetStream,
    tree: cKDTree,
    frame_id: int,
    scene: str,
    shape_hw: tuple[int, int],
    radius: float,
    cache_root: Path,
    use_cache: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = _vertex_map_cache_path(cache_root, scene, frame_id, radius)
    depth = stream.load_depth(frame_id)
    if depth.shape != shape_hw:
        raise ValueError(f"depth shape changed for frame {frame_id}: expected={shape_hw} got={depth.shape}")
    if use_cache and cache_path.exists():
        with np.load(cache_path) as payload:
            vertex_idx = np.asarray(payload["vertex_idx"], dtype=np.int64)
            diag = json.loads(str(payload["diag_json"]))
        if tuple(vertex_idx.shape) == tuple(shape_hw):
            diag["cache_hit"] = True
            diag["cache_path"] = _rel(cache_path)
            return vertex_idx, diag
    pose = stream.load_pose(frame_id)
    intrinsics = stream.load_intrinsics()
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(intrinsics)):
        valid_depth = int(np.count_nonzero(np.isfinite(depth) & (depth > 0)))
        vertex_idx = np.full(shape_hw, -1, dtype=np.int64)
        diag = {
            "frame_id": int(frame_id),
            "cache_hit": False,
            "cache_path": _rel(cache_path),
            "valid_depth_pixels": valid_depth,
            "nn_hit_pixels": 0,
            "nn_hit_ratio": 0.0,
            "nn_radius": float(radius),
            "nn_distance_mean": None,
            "nn_distance_median": None,
            "invalid_pose_or_intrinsics": True,
            "dropped_nonfinite_world_pixels": valid_depth,
        }
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                vertex_idx=vertex_idx.astype(np.int64, copy=False),
                diag_json=json.dumps(diag, sort_keys=True),
            )
        return vertex_idx, diag
    world, pixels = _world_points_from_depth(depth, intrinsics, pose)
    if world.shape[0] > 0:
        finite_world = np.all(np.isfinite(world), axis=1)
        dropped_nonfinite = int(world.shape[0] - np.count_nonzero(finite_world))
        if dropped_nonfinite:
            world = world[finite_world]
            pixels = pixels[finite_world]
    else:
        dropped_nonfinite = 0
    vertex_idx = np.full(shape_hw, -1, dtype=np.int64)
    if world.shape[0] > 0:
        try:
            distances, indices = tree.query(world, k=1, distance_upper_bound=float(radius), workers=-1)
        except TypeError:
            distances, indices = tree.query(world, k=1, distance_upper_bound=float(radius))
        valid = np.isfinite(distances) & (indices < int(tree.n))
        if np.any(valid):
            vertex_idx[pixels[valid, 0], pixels[valid, 1]] = indices[valid].astype(np.int64)
            valid_distances = distances[valid]
        else:
            valid_distances = np.asarray([], dtype=np.float32)
    else:
        valid = np.zeros((0,), dtype=bool)
        valid_distances = np.asarray([], dtype=np.float32)
    diag = {
        "frame_id": int(frame_id),
        "cache_hit": False,
        "cache_path": _rel(cache_path),
        "valid_depth_pixels": int(world.shape[0]),
        "nn_hit_pixels": int(np.count_nonzero(vertex_idx >= 0)),
        "nn_hit_ratio": float(np.count_nonzero(vertex_idx >= 0) / max(1, world.shape[0])),
        "nn_radius": float(radius),
        "nn_distance_mean": float(np.mean(valid_distances)) if valid_distances.size else None,
        "nn_distance_median": float(np.median(valid_distances)) if valid_distances.size else None,
        "invalid_pose_or_intrinsics": False,
        "dropped_nonfinite_world_pixels": int(dropped_nonfinite),
    }
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, vertex_idx=vertex_idx.astype(np.int64, copy=False), diag_json=json.dumps(diag, sort_keys=True))
    return vertex_idx, diag


def _stream3d_pred_2d(
    *,
    vertex_labels: np.ndarray,
    vertex_idx: np.ndarray,
) -> np.ndarray:
    pred = np.zeros(vertex_idx.shape, dtype=np.int64)
    valid = (vertex_idx >= 0) & (vertex_idx < vertex_labels.shape[0])
    pred[valid] = vertex_labels[vertex_idx[valid]]
    return pred


class SparseSceneIoU:
    def __init__(self) -> None:
        self.pred_area: defaultdict[int, int] = defaultdict(int)
        self.gt_area: defaultdict[int, int] = defaultdict(int)
        self.intersection: defaultdict[tuple[int, int], int] = defaultdict(int)
        self.frame_count = 0
        self.pixel_count = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
        pred = np.asarray(pred, dtype=np.int64)
        gt = np.asarray(gt, dtype=np.int64)
        self.frame_count += 1
        self.pixel_count += int(pred.size)
        pred_pos = pred > 0
        gt_pos = gt > 0
        if np.any(pred_pos):
            ids, counts = np.unique(pred[pred_pos], return_counts=True)
            for value, count in zip(ids, counts):
                self.pred_area[int(value)] += int(count)
        if np.any(gt_pos):
            ids, counts = np.unique(gt[gt_pos], return_counts=True)
            for value, count in zip(ids, counts):
                self.gt_area[int(value)] += int(count)
        both = pred_pos & gt_pos
        if np.any(both):
            pred_vals = pred[both]
            gt_vals = gt[both]
            base = int(np.max(gt_vals)) + 1
            encoded = pred_vals * base + gt_vals
            ids, counts = np.unique(encoded, return_counts=True)
            for value, count in zip(ids, counts):
                self.intersection[(int(value // base), int(value % base))] += int(count)

    def build(self, *, min_pred_pixels: int, min_gt_pixels: int) -> dict[str, Any]:
        pred_ids_all = sorted(self.pred_area)
        gt_ids_all = sorted(self.gt_area)
        pred_ids = [pid for pid in pred_ids_all if self.pred_area[pid] >= int(min_pred_pixels)]
        gt_ids = [gid for gid in gt_ids_all if self.gt_area[gid] >= int(min_gt_pixels)]
        pred_index = {pid: idx for idx, pid in enumerate(pred_ids)}
        gt_index = {gid: idx for idx, gid in enumerate(gt_ids)}
        iou = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
        inter = np.zeros_like(iou)
        for (pid, gid), count in self.intersection.items():
            if pid not in pred_index or gid not in gt_index:
                continue
            pidx = pred_index[pid]
            gidx = gt_index[gid]
            union = int(self.pred_area[pid]) + int(self.gt_area[gid]) - int(count)
            if union > 0:
                inter[pidx, gidx] = float(count)
                iou[pidx, gidx] = float(count) / float(union)
        return {
            "pred_ids": pred_ids,
            "gt_ids": gt_ids,
            "iou": iou,
            "intersection": inter,
            "pred_area": np.asarray([self.pred_area[pid] for pid in pred_ids], dtype=np.int64),
            "gt_area": np.asarray([self.gt_area[gid] for gid in gt_ids], dtype=np.int64),
            "pred_ids_all_count": len(pred_ids_all),
            "gt_ids_all_count": len(gt_ids_all),
        }


def _max_cardinality_match_count(iou_subset: np.ndarray, threshold: float) -> int:
    if iou_subset.size == 0 or iou_subset.shape[0] == 0 or iou_subset.shape[1] == 0:
        return 0
    eligible = iou_subset >= float(threshold)
    if not np.any(eligible):
        return 0
    cost = np.where(eligible, 0, 1).astype(np.int8, copy=False)
    rows, cols = linear_sum_assignment(cost)
    return int(np.count_nonzero(cost[rows, cols] == 0))


def _ap_from_scores(iou: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    gt_count = int(iou.shape[1])
    pred_count = int(iou.shape[0])
    if gt_count == 0:
        return {"ap": None, "tp": 0, "fp": pred_count, "gt_count": 0, "precision": None, "recall": None}
    if pred_count == 0:
        return {"ap": 0.0, "tp": 0, "fp": 0, "gt_count": gt_count, "precision": 0.0, "recall": 0.0}
    if scores.shape[0] != pred_count:
        scores = np.ones((pred_count,), dtype=np.float32)
    unique_scores = np.unique(scores)[::-1]
    precision: list[float] = []
    recall: list[float] = []
    best_tp = 0
    best_fp = pred_count
    for score in unique_scores:
        included = np.nonzero(scores >= score)[0]
        tp = _max_cardinality_match_count(iou[included], threshold)
        fp = int(included.shape[0]) - tp
        precision.append(float(tp / max(1, tp + fp)))
        recall.append(float(tp / gt_count))
        if tp > best_tp or (tp == best_tp and fp < best_fp):
            best_tp = int(tp)
            best_fp = int(fp)
    mrec = np.asarray([0.0, *recall, 1.0], dtype=np.float64)
    mpre = np.asarray([0.0, *precision, 0.0], dtype=np.float64)
    for idx in range(mpre.size - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    changed = np.nonzero(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
    return {
        "ap": ap,
        "tp": int(best_tp),
        "fp": int(best_fp),
        "gt_count": gt_count,
        "precision": float(best_tp / max(1, best_tp + best_fp)),
        "recall": float(best_tp / gt_count),
        "score_threshold_count": int(unique_scores.shape[0]),
    }


def _score_free_match_metrics(iou: np.ndarray, threshold: float) -> dict[str, Any]:
    pred_count = int(iou.shape[0])
    gt_count = int(iou.shape[1])
    tp = _max_cardinality_match_count(iou, threshold)
    fp = pred_count - tp
    fn = gt_count - tp
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "pred_count": pred_count,
        "gt_count": gt_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "definition": "score-free max-cardinality one-to-one matching using all predictions at this IoU threshold",
    }


def _score_vector(pred_ids: list[int], pred_area: np.ndarray, *, score_mode: str, input_scores: np.ndarray | None) -> np.ndarray:
    if score_mode == "constant":
        return np.ones((len(pred_ids),), dtype=np.float32)
    if score_mode == "pred_area":
        area = np.asarray(pred_area, dtype=np.float32)
        if area.size == 0:
            return area
        return area / max(1.0, float(np.max(area)))
    if score_mode == "input" and input_scores is not None:
        out = np.ones((len(pred_ids),), dtype=np.float32)
        for idx, pid in enumerate(pred_ids):
            source_idx = int(pid) - 1
            if 0 <= source_idx < input_scores.shape[0]:
                out[idx] = float(input_scores[source_idx])
        return out
    raise ValueError(f"unsupported score_mode={score_mode}")


def _summarize_iou(
    *,
    accumulator: SparseSceneIoU,
    min_pred_pixels: int,
    min_gt_pixels: int,
    score_mode: str,
    input_scores: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray, list[int], list[int]]:
    built = accumulator.build(min_pred_pixels=min_pred_pixels, min_gt_pixels=min_gt_pixels)
    iou = np.asarray(built["iou"], dtype=np.float32)
    pred_ids = list(built["pred_ids"])
    gt_ids = list(built["gt_ids"])
    scores = _score_vector(pred_ids, np.asarray(built["pred_area"]), score_mode=score_mode, input_scores=input_scores)
    ap_by_threshold = {f"{threshold:.2f}": _ap_from_scores(iou, scores, threshold) for threshold in AP_THRESHOLDS}
    ap_values = [row["ap"] for row in ap_by_threshold.values() if row["ap"] is not None]
    score_unique_count = int(np.unique(scores).shape[0]) if scores.size else 0
    if iou.shape[1]:
        gt_best = np.max(iou, axis=0) if iou.shape[0] else np.zeros((iou.shape[1],), dtype=np.float32)
    else:
        gt_best = np.asarray([], dtype=np.float32)
    if iou.shape[0]:
        pred_best = np.max(iou, axis=1) if iou.shape[1] else np.zeros((iou.shape[0],), dtype=np.float32)
    else:
        pred_best = np.asarray([], dtype=np.float32)
    summary = {
        "ap": float(np.mean(ap_values)) if ap_values else None,
        "ap50": ap_by_threshold["0.50"]["ap"],
        "ap25": _ap_from_scores(iou, scores, 0.25)["ap"],
        "ap_thresholds": AP_THRESHOLDS,
        "ap_by_threshold": ap_by_threshold,
        "ap_integral": "score-threshold precision envelope; same-score predictions are matched together by max-cardinality bipartite matching",
        "score_mode": score_mode,
        "score_unique_count": score_unique_count,
        "score_protocol_note": (
            "All predictions have one score; AP has a single real operating point and is strongly affected by prediction count. "
            "Use score_free_match_at_025/050 and best-IoU diagnostics to interpret unscored outputs."
            if score_unique_count <= 1
            else "Predictions have multiple scores; AP depends on the supplied score ordering."
        ),
        "score_free_match_at_025": _score_free_match_metrics(iou, 0.25),
        "score_free_match_at_050": _score_free_match_metrics(iou, 0.50),
        "min_pred_pixels": int(min_pred_pixels),
        "min_gt_pixels": int(min_gt_pixels),
        "evaluated_pred_count": int(len(pred_ids)),
        "evaluated_gt_count": int(len(gt_ids)),
        "raw_pred_count": int(built["pred_ids_all_count"]),
        "raw_gt_count": int(built["gt_ids_all_count"]),
        "pred_area_total_pixels": int(sum(accumulator.pred_area.values())),
        "gt_area_total_pixels": int(sum(accumulator.gt_area.values())),
        "intersection_pair_count": int(len(accumulator.intersection)),
        "frame_count": int(accumulator.frame_count),
        "pixel_count": int(accumulator.pixel_count),
        "gt_best_iou_mean": float(np.mean(gt_best)) if gt_best.size else None,
        "gt_best_iou_median": float(np.median(gt_best)) if gt_best.size else None,
        "gt_best_iou_max": float(np.max(gt_best)) if gt_best.size else None,
        "pred_best_iou_mean": float(np.mean(pred_best)) if pred_best.size else None,
        "pred_best_iou_median": float(np.median(pred_best)) if pred_best.size else None,
        "pred_best_iou_max": float(np.max(pred_best)) if pred_best.size else None,
        "gt_recall_best_iou_ge_025": float(np.mean(gt_best >= 0.25)) if gt_best.size else None,
        "gt_recall_best_iou_ge_050": float(np.mean(gt_best >= 0.50)) if gt_best.size else None,
    }
    return summary, iou, pred_ids, gt_ids


def _top_iou_rows(iou: np.ndarray, pred_ids: list[int], gt_ids: list[int], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if iou.size == 0:
        return rows
    flat = np.argsort(iou.reshape(-1))[::-1]
    for flat_idx in flat[: int(top_k)]:
        pidx = int(flat_idx // max(1, iou.shape[1]))
        gidx = int(flat_idx % max(1, iou.shape[1]))
        value = float(iou[pidx, gidx])
        if value <= 0:
            break
        rows.append({"pred_id": int(pred_ids[pidx]), "gt_id": int(gt_ids[gidx]), "iou": value})
    return rows


def _run_one(
    *,
    scene: str,
    method: str,
    stride: int,
    output_root: Path,
    pipeline_root: Path,
    stream3d_config: str,
    score_mode: str,
    min_pred_pixels: int,
    min_gt_pixels: int,
    vertex_nn_radius: float,
    vertex_cache_root: Path,
    use_cache: bool,
    max_frames: int,
) -> dict[str, Any]:
    stream = ScanNetStream(scene, root=ROOT / "data" / "scannet" / "processed")
    frame_ids = stream.frame_ids(stride=stride, max_frames=max_frames if max_frames > 0 else None)
    if not frame_ids:
        raise RuntimeError(f"no frames selected for scene={scene} stride={stride}")
    shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
    accumulator = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    method_diag: dict[str, Any] = {}
    input_scores: np.ndarray | None = None
    if method == "soma":
        pipeline_summary = _read_json(pipeline_root / "pipeline_summary.json")
        mask_dir = _resolve_pipeline_mask_dir(scene=scene, pipeline_summary=pipeline_summary, override_mask_root=None)
        objectlet_variant = _best_objectlet_variant(pipeline_root, "best")
        _support_by_frame, mask_to_object_idx, support_diag = _load_pipeline_support(
            pipeline_root=pipeline_root,
            scene=scene,
            objectlet_variant=objectlet_variant,
            success_only=True,
        )
        method_diag = {
            "pipeline_root": _rel(pipeline_root),
            "pipeline_summary": _rel(pipeline_root / "pipeline_summary.json"),
            "pipeline_summary_sha256": _sha256(pipeline_root / "pipeline_summary.json"),
            "mask_dir": _rel(mask_dir),
            "mask_dir_source": "pipeline_summary.mask_frame_coverage.mask_dir",
            "objectlet_variant": objectlet_variant,
            "support_diag": support_diag,
            "input_condition_note": "SOMA 2D masks are read from the pipeline-resolved mask dir; evaluated stride is selected by frame_ids.",
        }
        for frame_id in frame_ids:
            gt = _load_gt_2d(scene, frame_id, shape_hw)
            pred, diag = _soma_pred_2d(mask_dir=mask_dir, frame_id=frame_id, shape_hw=shape_hw, mask_to_object_idx=mask_to_object_idx)
            accumulator.add(pred, gt)
            frame_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "stride": int(stride),
                    "frame_id": int(frame_id),
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                    **diag,
                }
            )
    elif method == "stream3d":
        scene_points, _scene_colors, mesh_path = _load_scene_mesh(scene)
        tree = cKDTree(scene_points)
        vertex_labels, input_scores, stream3d_diag = _load_stream3d_vertex_labels(
            scene=scene,
            config=stream3d_config,
            vertex_count=int(scene_points.shape[0]),
        )
        method_diag = {
            **stream3d_diag,
            "mesh_path": _rel(mesh_path),
            "mesh_vertex_count": int(scene_points.shape[0]),
            "projection_mode": "depth_pixel_to_nearest_scannet_mesh_vertex",
            "vertex_nn_radius": float(vertex_nn_radius),
            "vertex_cache_root": _rel(vertex_cache_root),
            "input_condition_note": "Stream3D prediction is a scene-level ScanNet mesh vertex mask; evaluated stride controls only the rendered frame set.",
        }
        for frame_id in frame_ids:
            gt = _load_gt_2d(scene, frame_id, shape_hw)
            vertex_idx, vertex_diag = _load_or_build_vertex_map(
                stream=stream,
                tree=tree,
                frame_id=frame_id,
                scene=scene,
                shape_hw=shape_hw,
                radius=vertex_nn_radius,
                cache_root=vertex_cache_root,
                use_cache=use_cache,
            )
            pred = _stream3d_pred_2d(vertex_labels=vertex_labels, vertex_idx=vertex_idx)
            accumulator.add(pred, gt)
            frame_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "stride": int(stride),
                    "frame_id": int(frame_id),
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                    **vertex_diag,
                }
            )
    else:
        raise ValueError(f"unsupported method={method}")
    summary, iou, pred_ids, gt_ids = _summarize_iou(
        accumulator=accumulator,
        min_pred_pixels=min_pred_pixels,
        min_gt_pixels=min_gt_pixels,
        score_mode=score_mode,
        input_scores=input_scores,
    )
    output_dir = output_root / f"{scene}_{method}_stride{stride}_{score_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_rows = _top_iou_rows(iou, pred_ids, gt_ids, top_k=50)
    payload = {
        "phase": "v65_scene_level_multiview_2d_ap",
        "scene": scene,
        "method": method,
        "stride": int(stride),
        "frame_count": int(len(frame_ids)),
        "frame_first": int(frame_ids[0]),
        "frame_last": int(frame_ids[-1]),
        "frame_id_hash": _hash_text(",".join(str(fid) for fid in frame_ids)),
        "pixel_grid": {"height": int(shape_hw[0]), "width": int(shape_hw[1]), "source": "ScanNet depth resolution"},
        "gt_source": _rel(ROOT / "data" / "scannet" / "processed" / scene / "instance" / "instance"),
        "gt_resize": "nearest_to_depth_resolution",
        "pred_resize_or_projection": "nearest_to_depth_resolution" if method == "soma" else "depth_pixel_to_nearest_mesh_vertex",
        "diagnostic_only": True,
        "metric_scope": "scene-level multi-view 2D object AP; full-scene matching over all evaluated frames",
        "matching_scope": "global_scene_not_per_view",
        "method_diag": method_diag,
        "summary": summary,
        "top_iou_pairs": top_rows,
        "outputs": {
            "summary_json": _rel(output_dir / "summary.json"),
            "frame_rows_csv": _rel(output_dir / "frame_rows.csv"),
            "top_iou_pairs_csv": _rel(output_dir / "top_iou_pairs.csv"),
        },
    }
    _write_csv(output_dir / "frame_rows.csv", frame_rows)
    _write_csv(output_dir / "top_iou_pairs.csv", top_rows)
    _write_json(output_dir / "summary.json", payload)
    payload["outputs"].update(
        {
            "frame_rows_csv_sha256": _sha256(output_dir / "frame_rows.csv"),
            "top_iou_pairs_csv_sha256": _sha256(output_dir / "top_iou_pairs.csv"),
            "sha256_note": "summary_json hash is recorded in the aggregate SHA256SUMS.txt sidecar to avoid self-referential hashes.",
        }
    )
    _write_json(output_dir / "summary.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    pipeline_root = _project(args.pipeline_root)
    vertex_cache_root = _project(args.vertex_cache_root)
    methods = _parse_csv_list(args.methods)
    strides = [int(value) for value in _parse_csv_list(args.strides)]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in methods:
        for stride in strides:
            payload = _run_one(
                scene=args.scene,
                method=method,
                stride=int(stride),
                output_root=output_root,
                pipeline_root=pipeline_root,
                stream3d_config=args.stream3d_config,
                score_mode=args.score_mode,
                min_pred_pixels=int(args.min_pred_pixels),
                min_gt_pixels=int(args.min_gt_pixels),
                vertex_nn_radius=float(args.vertex_nn_radius),
                vertex_cache_root=vertex_cache_root,
                use_cache=bool(args.use_cache),
                max_frames=int(args.max_frames),
            )
            summaries.append(payload)
            rows.append(
                {
                    "scene": payload["scene"],
                    "method": payload["method"],
                    "stride": payload["stride"],
                    "frame_count": payload["frame_count"],
                    "score_mode": payload["summary"]["score_mode"],
                    "AP": payload["summary"]["ap"],
                    "AP50": payload["summary"]["ap50"],
                    "AP25": payload["summary"]["ap25"],
                    "evaluated_pred_count": payload["summary"]["evaluated_pred_count"],
                    "evaluated_gt_count": payload["summary"]["evaluated_gt_count"],
                    "gt_best_iou_mean": payload["summary"]["gt_best_iou_mean"],
                    "gt_recall_best_iou_ge_025": payload["summary"]["gt_recall_best_iou_ge_025"],
                    "gt_recall_best_iou_ge_050": payload["summary"]["gt_recall_best_iou_ge_050"],
                    "summary_json": payload["outputs"]["summary_json"],
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    aggregate = {
        "phase": "v65_scene_level_multiview_2d_ap_aggregate",
        "scene": args.scene,
        "methods": methods,
        "strides": strides,
        "score_mode": args.score_mode,
        "min_pred_pixels": int(args.min_pred_pixels),
        "min_gt_pixels": int(args.min_gt_pixels),
        "vertex_nn_radius": float(args.vertex_nn_radius),
        "diagnostic_only": True,
        "note": "This diagnostic isolates scene-level multi-view mask grouping from 3D AP's D4RT geometry materialization; it is not the official ScanNet 3D AP.",
        "rows": rows,
    }
    _write_csv(output_root / "aggregate_rows.csv", rows)
    aggregate["outputs"] = {
        "aggregate_summary_json": _rel(output_root / "aggregate_summary.json"),
        "aggregate_rows_csv": _rel(output_root / "aggregate_rows.csv"),
        "sha256sums": _rel(output_root / "SHA256SUMS.txt"),
        "hash_policy": "Final file hashes are stored in SHA256SUMS.txt, outside the JSON files being hashed.",
    }
    _write_json(output_root / "aggregate_summary.json", aggregate)
    hash_files = [output_root / "aggregate_summary.json", output_root / "aggregate_rows.csv"]
    for payload in summaries:
        summary_path = _project(payload["outputs"]["summary_json"])
        frame_rows_path = _project(payload["outputs"]["frame_rows_csv"])
        top_pairs_path = _project(payload["outputs"]["top_iou_pairs_csv"])
        hash_files.extend([summary_path, frame_rows_path, top_pairs_path])
    _write_sha256sums(output_root / "SHA256SUMS.txt", hash_files)
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute diagnostic scene-level multi-view 2D AP for v65 SOMA/Stream3D.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--methods", default="soma,stream3d", help="comma-separated: soma,stream3d")
    parser.add_argument("--strides", default="5,10", help="comma-separated frame strides")
    parser.add_argument("--pipeline-root", required=True, help="v65 SOMA fullscene pipeline root")
    parser.add_argument("--stream3d-config", default="scannet")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-mode", choices=["constant", "pred_area", "input"], default="constant")
    parser.add_argument("--min-pred-pixels", type=int, default=1)
    parser.add_argument("--min-gt-pixels", type=int, default=1)
    parser.add_argument("--vertex-nn-radius", type=float, default=0.08)
    parser.add_argument("--vertex-cache-root", default="outputs/cache/v65_scene_multiview_vertex_maps")
    parser.add_argument("--use-cache", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
