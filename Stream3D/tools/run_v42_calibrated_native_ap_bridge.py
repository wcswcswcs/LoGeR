from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.sim3 import apply_sim3_to_xyz, fit_sim3_umeyama
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _parse_radii(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else None


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, float(q))) if values.size else None


def _load_native_rows(path: Path, *, scene: str, variant: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        if str(row.get("scene", "")) != scene:
            continue
        if str(row.get("variant", "")) != variant:
            continue
        if str(row.get("source", "")) != source:
            continue
        rows.append(
            {
                "scene": scene,
                "variant": variant,
                "source": source,
                "object_id": int(row["object_id"]),
                "tube_id": int(row["tube_id"]),
                "frame_id": int(row["frame_id"]),
                "xyz": np.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32),
                "uv": np.asarray([float(row["u"]), float(row["v"])], dtype=np.float32),
                "visibility": float(row.get("visibility", 0.0) or 0.0),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
            }
        )
    return rows


def _load_scene_points(scene: str) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for calibrated native AP bridge diagnostics") from exc
    stream = ScanNetStream(seq_name=scene)
    points = np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"failed to load scene mesh points for {scene}: {stream.mesh_path}")
    return points


def _calibration_anchors_for_frame(
    *,
    stream: ScanNetStream,
    tree: cKDTree,
    scene_points: np.ndarray,
    intrinsics: np.ndarray,
    frame_id: int,
    rows: list[dict[str, Any]],
    backproject_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    depth = stream.load_depth(frame_id)
    pose = stream.load_pose(frame_id)
    if not np.isfinite(pose).all():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    h, w = depth.shape
    xyz = np.stack([np.asarray(row["xyz"], dtype=np.float32) for row in rows], axis=0)
    uv = np.stack([np.asarray(row["uv"], dtype=np.float32) for row in rows], axis=0)
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    x_valid = x[valid]
    y_valid = y[valid]
    z = depth[y_valid, x_valid]
    depth_valid = np.isfinite(z) & (z > 0.0)
    if not np.any(depth_valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    x_float = x_valid[depth_valid].astype(np.float32)
    y_float = y_valid[depth_valid].astype(np.float32)
    z_float = z[depth_valid].astype(np.float32)
    src = xyz[valid][depth_valid].astype(np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    cam = np.stack(
        [
            (x_float - cx) * z_float / fx,
            (y_float - cy) * z_float / fy,
            z_float,
            np.ones_like(z_float),
        ],
        axis=1,
    )
    world = (pose @ cam.T).T[:, :3].astype(np.float32)
    finite_world = np.isfinite(world).all(axis=1)
    if not np.any(finite_world):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    src = src[finite_world]
    world = world[finite_world]
    dist, idx = tree.query(world, k=1, distance_upper_bound=float(backproject_radius))
    hit = np.isfinite(dist) & (idx < scene_points.shape[0])
    if not np.any(hit):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return src[hit].astype(np.float32), scene_points[idx[hit].astype(np.int64)].astype(np.float32), dist[hit].astype(np.float32)


def _build_calibration_anchors(
    *,
    scene: str,
    rows: list[dict[str, Any]],
    scene_points: np.ndarray,
    tree: cKDTree,
    min_visibility: float,
    min_confidence: float,
    backproject_radius: float,
    max_anchors: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stream = ScanNetStream(seq_name=scene)
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    intrinsics = stream.load_intrinsics()
    candidates = [
        row
        for row in rows
        if float(row["visibility"]) >= float(min_visibility)
        and float(row["confidence"]) >= float(min_confidence)
        and np.isfinite(np.asarray(row["xyz"], dtype=np.float32)).all()
        and np.isfinite(np.asarray(row["uv"], dtype=np.float32)).all()
    ]
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_frame[int(row["frame_id"])].append(row)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []
    for frame_id in sorted(by_frame):
        src, dst, dist = _calibration_anchors_for_frame(
            stream=stream,
            tree=tree,
            scene_points=scene_points,
            intrinsics=intrinsics,
            frame_id=int(frame_id),
            rows=by_frame[int(frame_id)],
            backproject_radius=float(backproject_radius),
        )
        if src.size:
            src_parts.append(src)
            dst_parts.append(dst)
            dist_parts.append(dist)
    if src_parts:
        src_all = np.concatenate(src_parts, axis=0)
        dst_all = np.concatenate(dst_parts, axis=0)
        dist_all = np.concatenate(dist_parts, axis=0)
    else:
        src_all = np.empty((0, 3), dtype=np.float32)
        dst_all = np.empty((0, 3), dtype=np.float32)
        dist_all = np.empty((0,), dtype=np.float32)
    if int(max_anchors) > 0 and src_all.shape[0] > int(max_anchors):
        keep = np.linspace(0, src_all.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
        src_all = src_all[keep]
        dst_all = dst_all[keep]
        dist_all = dist_all[keep]
    diag = {
        "scene": scene,
        "native_row_count": int(len(rows)),
        "calibration_candidate_count": int(len(candidates)),
        "calibration_anchor_count": int(src_all.shape[0]),
        "calibration_anchor_hit_rate": float(src_all.shape[0] / max(len(candidates), 1)),
        "calibration_backproject_radius": float(backproject_radius),
        "calibration_backproject_distance_mean": _safe_mean(dist_all),
        "calibration_backproject_distance_p90": _safe_quantile(dist_all, 0.90),
        "calibration_frame_count": int(len(by_frame)),
        "calibration_frame_min": int(min(by_frame)) if by_frame else None,
        "calibration_frame_max": int(max(by_frame)) if by_frame else None,
    }
    return src_all, dst_all, diag


def _fit_robust_sim3(
    source: np.ndarray,
    target: np.ndarray,
    *,
    trim_quantile: float = 0.80,
    min_anchors: int = 16,
) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have shape (N, 3)")
    if source.shape[0] < int(min_anchors):
        raise ValueError(f"at least {int(min_anchors)} anchors are required")
    first = fit_sim3_umeyama(source, target)
    first_pred = apply_sim3_to_xyz(source, transform=first).astype(np.float64)
    first_residual = np.linalg.norm(first_pred - target, axis=1)
    threshold = float(np.quantile(first_residual, float(trim_quantile))) if first_residual.size else float("nan")
    keep = np.isfinite(first_residual) & (first_residual <= threshold)
    if int(np.count_nonzero(keep)) >= int(min_anchors):
        final = fit_sim3_umeyama(source[keep], target[keep])
        fit_source = source[keep]
        fit_target = target[keep]
        kept_count = int(np.count_nonzero(keep))
    else:
        final = first
        fit_source = source
        fit_target = target
        kept_count = int(source.shape[0])
    final_pred = apply_sim3_to_xyz(fit_source, transform=final).astype(np.float64)
    final_residual = np.linalg.norm(final_pred - fit_target, axis=1)
    return {
        "scale": float(final["scale"]),
        "rot": np.asarray(final["rot"], dtype=np.float64),
        "trans": np.asarray(final["trans"], dtype=np.float64),
        "rotation_det": float(np.linalg.det(np.asarray(final["rot"], dtype=np.float64))),
        "initial_anchor_count": int(source.shape[0]),
        "kept_anchor_count": int(kept_count),
        "trim_quantile": float(trim_quantile),
        "trim_threshold": threshold,
        "initial_residual_mean": _safe_mean(first_residual),
        "initial_residual_p50": _safe_quantile(first_residual, 0.50),
        "initial_residual_p90": _safe_quantile(first_residual, 0.90),
        "final_residual_mean": _safe_mean(final_residual),
        "final_residual_p50": _safe_quantile(final_residual, 0.50),
        "final_residual_p90": _safe_quantile(final_residual, 0.90),
    }


def _object_vertices_from_transformed_points(
    *,
    rows: list[dict[str, Any]],
    transform: dict[str, Any],
    tree: cKDTree,
    scene_point_count: int,
    radius: float,
) -> tuple[dict[int, np.ndarray], dict[int, float], dict[str, Any]]:
    by_object: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        by_object[int(row["object_id"])].append(np.asarray(row["xyz"], dtype=np.float32))
    object_vertices: dict[int, np.ndarray] = {}
    scores: dict[int, float] = {}
    query_count = 0
    hit_count = 0
    distance_values: list[float] = []
    for object_id, points_list in sorted(by_object.items()):
        points = np.stack(points_list, axis=0).astype(np.float32) if points_list else np.empty((0, 3), dtype=np.float32)
        transformed = apply_sim3_to_xyz(points, transform=transform).astype(np.float32)
        query_count += int(transformed.shape[0])
        if transformed.size == 0:
            object_vertices[int(object_id)] = np.empty((0,), dtype=np.int64)
            scores[int(object_id)] = 0.0
            continue
        valid_points = np.isfinite(transformed).all(axis=1)
        transformed = transformed[valid_points]
        if transformed.size == 0:
            object_vertices[int(object_id)] = np.empty((0,), dtype=np.int64)
            scores[int(object_id)] = 0.0
            continue
        dist, idx = tree.query(transformed, k=1, distance_upper_bound=float(radius))
        valid = np.isfinite(dist) & (idx < int(scene_point_count))
        hit_count += int(np.count_nonzero(valid))
        distance_values.extend(float(v) for v in dist[valid].tolist())
        object_vertices[int(object_id)] = np.unique(idx[valid].astype(np.int64))
        scores[int(object_id)] = float(np.count_nonzero(valid))
    diag = {
        "native_point_query_count": int(query_count),
        "native_point_hit_count": int(hit_count),
        "calibrated_native_hit_rate": float(hit_count / max(query_count, 1)),
        "nn_distance_mean": float(np.mean(np.asarray(distance_values, dtype=np.float64))) if distance_values else None,
        "nn_distance_p90": float(np.quantile(np.asarray(distance_values, dtype=np.float64), 0.90)) if distance_values else None,
    }
    return object_vertices, scores, diag


def _write_prediction(
    *,
    scene: str,
    config: str,
    object_vertices: dict[int, np.ndarray],
    scores: dict[int, float],
    scene_point_count: int,
    native_point_rows: Path,
    radius: float,
    calibration_summary: dict[str, Any],
) -> dict[str, Any]:
    pred_dir = ROOT / "data/prediction" / f"{config}_class_agnostic"
    tmp_dir = ROOT / "data/TMP" / config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    kept = [(object_id, ids) for object_id, ids in sorted(object_vertices.items()) if np.asarray(ids).size > 0]
    masks = np.zeros((int(scene_point_count), len(kept)), dtype=bool)
    pred_scores = np.zeros((len(kept),), dtype=np.float32)
    for col, (object_id, ids_raw) in enumerate(kept):
        ids = np.asarray(ids_raw, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < int(scene_point_count))]
        masks[ids, col] = True
        pred_scores[col] = float(scores.get(int(object_id), float(ids.shape[0])))
    np.savez_compressed(
        pred_dir / f"{scene}.npz",
        pred_masks=masks,
        pred_score=pred_scores,
        pred_classes=np.zeros((len(kept),), dtype=np.int32),
    )
    pre_points = np.flatnonzero(np.any(masks, axis=1)).astype(np.int64)
    np.save(tmp_dir / f"{scene}_pre_points.npy", pre_points)
    manifest = build_prediction_manifest(
        output_config=config,
        root=ROOT,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(native_point_rows)],
        pre_points_policy="rgbd_pose_sim3_calibrated_native_xyz_nearest_mesh_vertex",
        support_policy="d4rt_native_xyz_sim3_nn_radius",
        notes=(
            "v42 calibrated native D4RT AP bridge diagnostic. Forbidden for method tables because "
            "the Sim3 calibration uses ScanNet RGB-D, pose, and mesh backprojection."
        ),
        extra={
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "alignment_source": "rgbd_pose_mesh_sim3_calibration_from_native_uv",
            "alignment_used_for_prediction": True,
            "phase": "v42_calibrated_native_ap_bridge",
            "nn_radius": float(radius),
            "calibration": calibration_summary,
        },
    )
    write_prediction_manifest(config, manifest, root=ROOT, pred_suffix="class_agnostic")
    return {
        "num_predictions": int(len(kept)),
        "num_scene_points": int(scene_point_count),
        "num_exported_points": int(pre_points.shape[0]),
        "mesh_coverage": float(pre_points.shape[0] / max(int(scene_point_count), 1)),
        "prediction_path": str(pred_dir / f"{scene}.npz"),
        "pre_points_path": str(tmp_dir / f"{scene}_pre_points.npy"),
    }


def _run_eval(config: str, output_root: Path) -> dict[str, Any]:
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{config}_class_agnostic.txt"
    log_path = eval_dir / f"{config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(ROOT / "data/prediction" / f"{config}_class_agnostic"),
        "--gt_path",
        str(ROOT / "data/scannet/gt"),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(ROOT / "data/TMP"),
        "--tmp_config",
        config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    metrics = _parse_metric_file(metric_file) if metric_file.exists() else {}
    return {
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "metric_file": str(metric_file),
        "log_path": str(log_path),
        "metrics": metrics,
    }


def _scene_quality(scene: str, config: str, radius: float) -> dict[str, Any]:
    pred_path = ROOT / "data/prediction" / f"{config}_class_agnostic" / f"{scene}.npz"
    pre_path = ROOT / "data/TMP" / config / f"{scene}_pre_points.npy"
    gt_path = ROOT / "data/scannet/gt" / f"{scene}.txt"
    with np.load(pred_path) as pred:
        masks = np.asarray(pred["pred_masks"], dtype=bool)
        scores = np.asarray(pred["pred_score"], dtype=np.float32)
    pre_points = np.load(pre_path).astype(np.int64)
    gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
    stats = _candidate_oracle_stats(masks, scores, gt_eval, pre_points)
    row = _quality_stats(scene, f"calibrated_native_nn_r{radius:g}", masks, scores, gt_eval, pre_points, stats)
    row["nn_radius"] = float(radius)
    return row


def _aggregate(rows: list[dict[str, Any]], eval_result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"scene_count": int(len(rows))}
    keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key != "scene"
    )
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.integer, np.floating))]
        if values:
            out[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
    out.update(eval_result.get("metrics", {}))
    out["eval_exit_code"] = int(eval_result.get("exit_code", -1))
    return out


def _finite_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        current = float(value)
    except (TypeError, ValueError):
        return None
    return current if np.isfinite(current) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="v42 calibrated native D4RT xyz AP bridge diagnostic.")
    parser.add_argument("--native-point-rows", default="outputs/audit/v42_streaming_memory_unioncap320_allframe_r1/memory_native_point_rows.csv")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--source", default="dinov2_maskcut")
    parser.add_argument("--radii", default="0.02,0.05,0.10,0.25,0.50")
    parser.add_argument("--calibration-backproject-radius", type=float, default=0.05)
    parser.add_argument("--calibration-trim-quantile", type=float, default=0.80)
    parser.add_argument("--calibration-min-anchors", type=int, default=64)
    parser.add_argument("--calibration-max-anchors", type=int, default=30000)
    parser.add_argument("--min-visibility", type=float, default=0.50)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--output-config-prefix", default="v42_calibrated_native_memory_allframe_r1")
    parser.add_argument("--output-root", default="outputs/audit/v42_calibrated_native_ap_bridge_allframe_r1")
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    radii = _parse_radii(str(args.radii))
    native_point_rows = ROOT / str(args.native_point_rows)
    output_root = ROOT / str(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_scene = {
        scene: _load_native_rows(native_point_rows, scene=scene, variant=str(args.variant), source=str(args.source))
        for scene in scenes
    }
    scene_points_by_scene: dict[str, np.ndarray] = {}
    trees_by_scene: dict[str, cKDTree] = {}
    transforms: dict[str, dict[str, Any]] = {}
    calibration_rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_points = _load_scene_points(scene)
        tree = cKDTree(scene_points)
        scene_points_by_scene[scene] = scene_points
        trees_by_scene[scene] = tree
        src, dst, anchor_diag = _build_calibration_anchors(
            scene=scene,
            rows=rows_by_scene[scene],
            scene_points=scene_points,
            tree=tree,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            backproject_radius=float(args.calibration_backproject_radius),
            max_anchors=int(args.calibration_max_anchors),
        )
        fit = _fit_robust_sim3(
            src,
            dst,
            trim_quantile=float(args.calibration_trim_quantile),
            min_anchors=int(args.calibration_min_anchors),
        )
        transforms[scene] = fit
        calibration_rows.append(
            {
                **anchor_diag,
                "scale": fit["scale"],
                "rotation_det": fit["rotation_det"],
                "kept_anchor_count": fit["kept_anchor_count"],
                "initial_residual_mean": fit["initial_residual_mean"],
                "initial_residual_p50": fit["initial_residual_p50"],
                "initial_residual_p90": fit["initial_residual_p90"],
                "final_residual_mean": fit["final_residual_mean"],
                "final_residual_p50": fit["final_residual_p50"],
                "final_residual_p90": fit["final_residual_p90"],
                "uses_gt_for_prediction": False,
                "uses_rgbd_for_prediction": True,
                "uses_pose_for_prediction": True,
                "uses_scannet_mesh_for_prediction": True,
                "forbidden_for_method_table": True,
            }
        )

    matrix: list[dict[str, Any]] = []
    scene_rows_all: list[dict[str, Any]] = []
    evals: dict[str, dict[str, Any]] = {}
    for radius in radii:
        radius_tag = str(f"{float(radius):.3f}").replace(".", "p")
        config = f"{args.output_config_prefix}_r{radius_tag}"
        scene_rows: list[dict[str, Any]] = []
        for scene in scenes:
            scene_points = scene_points_by_scene[scene]
            object_vertices, scores, hit_diag = _object_vertices_from_transformed_points(
                rows=rows_by_scene[scene],
                transform=transforms[scene],
                tree=trees_by_scene[scene],
                scene_point_count=int(scene_points.shape[0]),
                radius=float(radius),
            )
            write_diag = _write_prediction(
                scene=scene,
                config=config,
                object_vertices=object_vertices,
                scores=scores,
                scene_point_count=int(scene_points.shape[0]),
                native_point_rows=native_point_rows,
                radius=float(radius),
                calibration_summary={
                    "scene": scene,
                    "scale": transforms[scene]["scale"],
                    "rotation_det": transforms[scene]["rotation_det"],
                    "kept_anchor_count": transforms[scene]["kept_anchor_count"],
                    "final_residual_mean": transforms[scene]["final_residual_mean"],
                    "final_residual_p90": transforms[scene]["final_residual_p90"],
                },
            )
            quality = _scene_quality(scene, config, float(radius))
            row = {
                **quality,
                **write_diag,
                **hit_diag,
                "config": config,
                "scene": scene,
                "uses_gt_for_prediction": False,
                "uses_rgbd_for_prediction": True,
                "uses_pose_for_prediction": True,
                "uses_scannet_mesh_for_prediction": True,
                "forbidden_for_method_table": True,
            }
            scene_rows.append(row)
            scene_rows_all.append(row)
        eval_result = _run_eval(config, output_root)
        evals[config] = eval_result
        agg = _aggregate(scene_rows, eval_result)
        agg.update({"config": config, "nn_radius": float(radius)})
        matrix.append(agg)

    finite_ap_rows = [row for row in matrix if _finite_metric(row, "AP") is not None]
    if finite_ap_rows:
        best = max(finite_ap_rows, key=lambda row: float(row.get("AP") or -1.0))
        status = "OK_CALIBRATED_NATIVE_DIAGNOSTIC_AP_COMPUTED"
    elif matrix:
        best = max(matrix, key=lambda row: float(row.get("calibrated_native_hit_rate") or -1.0))
        status = "NO_GO_CALIBRATED_NATIVE_AP_NO_VALID_AP"
    else:
        best = {}
        status = "NO_GO_CALIBRATED_NATIVE_DIAGNOSTIC_EMPTY"
    summary = {
        "phase": "v42_calibrated_native_ap_bridge",
        "status": status,
        "native_point_rows": str(native_point_rows),
        "radii": radii,
        "calibration_rows": calibration_rows,
        "matrix": matrix,
        "scene_rows": scene_rows_all,
        "evals": evals,
        "best_by_AP": best,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "forbidden_for_method_table": True,
        "phase8_gate_pass": False,
        "phase8_gate_blocker": (
            "calibrated native bridge uses RGB-D/pose/mesh Sim3 calibration and is diagnostic-only, "
            "not method-compatible AP"
        ),
    }
    _write_json(output_root / "calibrated_native_ap_bridge_summary.json", summary)
    _write_csv(output_root / "calibrated_native_ap_bridge_matrix.csv", matrix)
    _write_csv(output_root / "calibrated_native_ap_bridge_scene_rows.csv", scene_rows_all)
    _write_csv(output_root / "calibrated_native_calibration_rows.csv", calibration_rows)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "status": summary["status"],
                    "best_config": best.get("config"),
                    "best_radius": best.get("nn_radius"),
                    "best_AP": best.get("AP"),
                    "best_AP50": best.get("AP50"),
                    "best_AP25": best.get("AP25"),
                    "best_calibrated_hit_rate": best.get("calibrated_native_hit_rate"),
                    "phase8_gate_pass": summary["phase8_gate_pass"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
