from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stream4d.scannet_stream import ScanNetStream


PROBE5_HARD_SCENES = {"scene0030_00", "scene0081_01"}
DIAG_MANIFEST = {
    "is_method_result": False,
    "is_diagnostic_only": True,
    "forbidden_for_method_table": True,
    "uses_scannet_depth_for_metric": True,
    "uses_scannet_pose_for_metric": True,
    "uses_scannet_mesh_for_metric": False,
    "uses_gt_instance_for_metric": True,
    "uses_gt_scale_or_sim3_for_prediction": False,
}


@dataclass(frozen=True)
class SourceRow:
    path: Path
    row: dict[str, Any]


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


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _variant_rows(v22_audit_root: Path) -> dict[str, SourceRow]:
    paths = [
        v22_audit_root / "v22_direct_reconstruction_probe5_r0_r8_raw" / "direct_reconstruction_summary.csv",
        v22_audit_root / "v22_direct_reconstruction_probe5_r4_eval_scene" / "direct_reconstruction_summary.csv",
        v22_audit_root / "v22_5_direct_xyz_local_transform_probe5" / "direct_reconstruction_summary.csv",
        v22_audit_root / "v22_6_eval_sim3_xyz_local_transform_probe5" / "direct_reconstruction_summary.csv",
        v22_audit_root / "v22_7_ref0_pose_scale_probe5" / "direct_reconstruction_summary.csv",
        v22_audit_root / "v22_8_ref0_local_scale_probe5" / "direct_reconstruction_summary.csv",
        v22_audit_root / "v22_9_ref0_pose_trajectory_scale_probe5" / "direct_reconstruction_summary.csv",
    ]
    out: dict[str, SourceRow] = {}
    for path in paths:
        for row in _read_csv(path):
            variant = row.get("variant")
            if variant and variant not in out:
                out[variant] = SourceRow(path=path, row=row)
    return out


def _scene_rows_by_variant(v22_audit_root: Path) -> dict[str, list[dict[str, str]]]:
    paths = [
        v22_audit_root / "v22_direct_reconstruction_probe5_r0_r8_raw" / "direct_reconstruction_scene_rows.csv",
        v22_audit_root / "v22_direct_reconstruction_probe5_r4_eval_scene" / "direct_reconstruction_scene_rows.csv",
        v22_audit_root / "v22_5_direct_xyz_local_transform_probe5" / "direct_reconstruction_scene_rows.csv",
        v22_audit_root / "v22_6_eval_sim3_xyz_local_transform_probe5" / "direct_reconstruction_scene_rows.csv",
        v22_audit_root / "v22_7_ref0_pose_scale_probe5" / "direct_reconstruction_scene_rows.csv",
        v22_audit_root / "v22_8_ref0_local_scale_probe5" / "direct_reconstruction_scene_rows.csv",
        v22_audit_root / "v22_9_ref0_pose_trajectory_scale_probe5" / "direct_reconstruction_scene_rows.csv",
        v22_audit_root / "v22_direct_reconstruction_scene0050_r1r2r3" / "direct_reconstruction_scene_rows.csv",
    ]
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        for row in _read_csv(path):
            variant = row.get("variant")
            if variant:
                merged = dict(row)
                merged["source_artifact"] = str(path)
                out[variant].append(merged)
    return out


def _metric(row: dict[str, Any], prefix: str, key: str) -> float | None:
    return _to_float(row.get(f"{prefix}_{key}"))


def _depth_row(
    name: str,
    source_variant: str,
    source: SourceRow,
    prefix: str,
    alignment_source: str,
    note: str,
) -> dict[str, Any]:
    row = source.row
    return {
        "variant": name,
        "source_variant": source_variant,
        "source_label": row.get("label", ""),
        "source_artifact": str(source.path),
        "alignment_source": alignment_source,
        "uses_eval_alignment": alignment_source not in {"none"},
        "AbsRel": _metric(row, prefix, "absrel"),
        "SqRel": _metric(row, prefix, "sqrel"),
        "RMSE": _metric(row, prefix, "rmse"),
        "RMSE_log": _metric(row, prefix, "rmse_log"),
        "MAE": _metric(row, prefix, "mae"),
        "delta1": _metric(row, prefix, "delta1"),
        "delta2": _metric(row, prefix, "delta2"),
        "delta3": _metric(row, prefix, "delta3"),
        "valid_pixel_ratio": _metric(row, prefix, "valid_pixel_ratio"),
        "median_scale": _to_float(row.get("depth_scale_median")),
        "linear_scale": _to_float(row.get("depth_ls_scale")),
        "linear_shift": _to_float(row.get("depth_ls_shift")),
        "note": note,
    }


def build_depth_tables(v22_audit_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    variants = _variant_rows(v22_audit_root)
    specs = [
        ("D0", "R16", "depth_raw", "none", "xyz_local raw z"),
        ("D1", "R16", "depth_raw", "none", "raw z metrics already reject non-positive z; positive-z-only dense map unavailable from sparse carrier cache"),
        ("D2", "R16", "depth_median", "depth_calibration", "xyz_local median-scale aligned to ScanNet depth"),
        ("D3", "R16", "depth_ls", "depth_calibration", "xyz_local linear scale-shift aligned to ScanNet depth"),
        ("D4", "R17", "depth_raw", "none", "xyz_local signed_log1p z"),
        ("D5", "R17", "depth_median", "depth_calibration", "signed_log1p median-scale aligned to ScanNet depth"),
        ("D6", "R18", "depth_raw", "none", "uv_pred + xyz_local z pinhole depth"),
        ("D7", "R22", "depth_raw", "ref0_pose", "xyz_ref0 through ScanNet ref0 pose"),
        ("D8", "R23", "depth_raw", "ref0_pose_eval_scale", "R23-style ref0 pose + eval-only scale depth"),
        ("D9", "R27", "depth_raw", "pose_trajectory", "pose-trajectory scale proxy depth"),
    ]
    rows = []
    for name, source_variant, prefix, alignment, note in specs:
        source = variants.get(source_variant)
        if source is None:
            rows.append({"variant": name, "source_variant": source_variant, "status": "missing_source_artifact", "note": note})
            continue
        cur = _depth_row(name, source_variant, source, prefix, alignment, note)
        cur["status"] = "ok" if cur.get("AbsRel") is not None else "missing_metric"
        rows.append(cur)
    _write_csv(out_dir / "depth_summary.csv", rows)
    _write_json(out_dir / "depth_summary.json", rows)
    _write_json(out_dir / "manifest.json", {**DIAG_MANIFEST, "alignment_source": "mixed", "phase": "v23_depth"})
    return rows


def _point_row(
    name: str,
    source_variant: str,
    source: SourceRow | None,
    *,
    metric_prefix: str,
    alignment_source: str,
    scope: str,
    note: str,
) -> dict[str, Any]:
    if source is None:
        return {"variant": name, "source_variant": source_variant, "status": "missing_source_artifact", "scope": scope, "note": note}
    row = source.row
    prefix = f"{metric_prefix}_" if metric_prefix else ""
    return {
        "variant": name,
        "source_variant": source_variant,
        "source_label": row.get("label", ""),
        "source_artifact": str(source.path),
        "scope": scope,
        "alignment_source": alignment_source,
        "uses_eval_alignment": alignment_source not in {"none", "self_sim3"},
        "Chamfer_L1": _to_float(row.get(f"{prefix}chamfer_l1")),
        "Chamfer_L2": _to_float(row.get(f"{prefix}chamfer_l2")),
        "Precision@1cm": _to_float(row.get(f"{prefix}precision@1cm")),
        "Precision@5cm": _to_float(row.get(f"{prefix}precision@5cm")),
        "Precision@10cm": _to_float(row.get(f"{prefix}precision@10cm")),
        "Precision@20cm": _to_float(row.get(f"{prefix}precision@20cm")),
        "Recall@1cm": _to_float(row.get(f"{prefix}recall@1cm")),
        "Recall@5cm": _to_float(row.get(f"{prefix}recall@5cm")),
        "Recall@10cm": _to_float(row.get(f"{prefix}recall@10cm")),
        "Recall@20cm": _to_float(row.get(f"{prefix}recall@20cm")),
        "F@1cm": _to_float(row.get(f"{prefix}fscore@1cm")),
        "F@5cm": _to_float(row.get(f"{prefix}fscore@5cm")),
        "F@10cm": _to_float(row.get(f"{prefix}fscore@10cm")),
        "F@20cm": _to_float(row.get(f"{prefix}fscore@20cm")),
        "Outlier@20cm": _to_float(row.get(f"{prefix}outlier_rate_20cm")),
        "Outlier@50cm": _to_float(row.get(f"{prefix}outlier_rate_50cm")),
        "pred_point_count": _to_float(row.get(f"{prefix}pred_point_count")),
        "gt_point_count": _to_float(row.get(f"{prefix}gt_point_count")),
        "per_instance_covered_gt_ratio": _to_float(row.get("per_instance_covered_gt_ratio")),
        "status": row.get("status", "ok"),
        "note": note,
    }


def build_pointcloud_tables(v22_audit_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    variants = _variant_rows(v22_audit_root)
    scene0050 = {}
    scene_path = v22_audit_root / "v22_direct_reconstruction_scene0050_r1r2r3" / "direct_reconstruction_summary.csv"
    for row in _read_csv(scene_path):
        if row.get("variant"):
            scene0050[row["variant"]] = SourceRow(scene_path, row)
    specs = [
        ("P0", "R16", variants.get("R16"), "camera_space", "none", "probe5", "xyz_local raw camera-space point cloud"),
        ("P1", "R17", variants.get("R17"), "camera_space", "none", "probe5", "xyz_local signed_log1p camera-space point cloud"),
        ("P2", "R18", variants.get("R18"), "camera_space", "none", "probe5", "uv_pred + local_z camera-space point cloud"),
        ("P3", "R19", variants.get("R19"), "camera_space", "none", "probe5", "uv_pred + signed_log1p(local_z) camera-space point cloud"),
        ("P4", "R22", variants.get("R22"), "", "ref0_pose", "probe5", "xyz_ref0 raw + ScanNet ref0 pose world point cloud"),
        ("P5", "R23", variants.get("R23"), "", "ref0_pose_eval_scale", "probe5", "xyz_ref0 + ref0 pose + eval-only scale"),
        ("P6", "R4", variants.get("R4"), "", "eval_sim3", "probe5", "full eval-Sim3 upper bound"),
        ("P7", "R1", scene0050.get("R1"), "", "none", "scene0050_only", "sliding-window raw world point cloud"),
        ("P8", "R2", scene0050.get("R2"), "", "self_sim3", "scene0050_only", "sliding-window self-Sim3 stitched point cloud"),
        ("P9", "R3", scene0050.get("R3"), "", "self_sim3", "scene0050_only", "scale-normalized self-Sim3 point cloud"),
        ("P10", "R27", variants.get("R27"), "", "pose_trajectory", "probe5", "pose-trajectory scale proxy point cloud"),
    ]
    rows = [_point_row(*spec[:3], metric_prefix=spec[3], alignment_source=spec[4], scope=spec[5], note=spec[6]) for spec in specs]
    _write_csv(out_dir / "pointcloud_summary.csv", rows)
    _write_json(out_dir / "pointcloud_summary.json", rows)
    scene_rows = []
    by_variant = _scene_rows_by_variant(v22_audit_root)
    for v in ("R16", "R17", "R18", "R19", "R22", "R23", "R4", "R27", "R1", "R2", "R3"):
        scene_rows.extend(by_variant.get(v, []))
    _write_csv(out_dir / "pointcloud_scene_rows.csv", scene_rows)
    _write_json(out_dir / "manifest.json", {**DIAG_MANIFEST, "alignment_source": "mixed", "phase": "v23_pointcloud"})
    return rows


def _backproject_pixel(x: float, y: float, z: float, intr: np.ndarray) -> np.ndarray:
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    return np.asarray([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=np.float64)


def _project_camera(point: np.ndarray, intr: np.ndarray) -> tuple[float, float] | None:
    if point[2] <= 1e-6 or not np.isfinite(point).all():
        return None
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    return float(point[0] * fx / point[2] + cx), float(point[1] * fy / point[2] + cy)


def _summarize_track_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0, "status": "empty"}
    epe = np.asarray([s["epe"] for s in samples if s.get("pseudo_visible")], dtype=np.float64)
    vis_gt = np.asarray([bool(s["pseudo_visible"]) for s in samples], dtype=bool)
    vis_pred = np.asarray([bool(s["pred_visible"]) for s in samples], dtype=bool)
    tp = int(np.count_nonzero(vis_gt & vis_pred))
    fp = int(np.count_nonzero(~vis_gt & vis_pred))
    fn = int(np.count_nonzero(vis_gt & ~vis_pred))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    out: dict[str, Any] = {
        "status": "ok",
        "sample_count": int(len(samples)),
        "visible_sample_count": int(epe.size),
        "EPE": float(np.mean(epe)) if epe.size else None,
        "EPE_median": float(np.median(epe)) if epe.size else None,
        "EPE_p90": float(np.percentile(epe, 90)) if epe.size else None,
        "PCK@1px": float(np.mean(epe < 1.0)) if epe.size else None,
        "PCK@3px": float(np.mean(epe < 3.0)) if epe.size else None,
        "PCK@5px": float(np.mean(epe < 5.0)) if epe.size else None,
        "PCK@10px": float(np.mean(epe < 10.0)) if epe.size else None,
        "visibility_precision": precision,
        "visibility_recall": recall,
        "visibility_F1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "out_of_frame_false_positive_rate": float(
            np.mean([bool(s["pred_visible"]) and not bool(s["gt_in_frame"]) for s in samples])
        ),
    }
    return out


def build_track_tables(
    seq_list: list[str],
    cache_root: Path,
    out_dir: Path,
    *,
    backbone: str,
    max_points_per_frame: int,
    depth_consistency_m: float,
    visibility_threshold: float,
) -> list[dict[str, Any]]:
    frame_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    controls = ("real_d4rt", "shuffle_target_frames", "shuffle_source_points", "source_uv_no_motion", "random_same_count")
    rng = np.random.default_rng(23)
    for scene in seq_list:
        scene_dir = cache_root / scene
        carrier_path = scene_dir / "carriers_window000.npz"
        manifest_path = scene_dir / "carriers_window000_manifest.json"
        if not carrier_path.exists() or not manifest_path.exists():
            for control in controls:
                scene_rows.append({"scene": scene, "control": control, "status": "missing_cache"})
            continue
        batch = np.load(carrier_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_ids = [int(v) for v in manifest.get("raw_frame_ids") or manifest.get("frame_indices") or []]
        if not frame_ids:
            frame_ids = list(range(int(batch["uv_pred"].shape[0])))
        stream = ScanNetStream(scene, backbone=backbone)
        intr = stream.load_intrinsics()
        uv_pred = np.asarray(batch["uv_pred"], dtype=np.float64)
        visibility = np.asarray(batch["visibility_prob"], dtype=np.float64)
        confidence = np.asarray(batch["confidence_prob"], dtype=np.float64)
        valid = np.asarray(batch["valid"], dtype=bool)
        src_xy = np.asarray(batch["src_xy"], dtype=np.int64)
        src_frame_global = np.asarray(batch["src_frame_global"], dtype=np.int64)
        src_uv = np.asarray(batch["src_uv"], dtype=np.float64)
        num_frames, num_points = uv_pred.shape[:2]
        point_indices = np.arange(num_points, dtype=np.int64)
        if point_indices.size > int(max_points_per_frame):
            point_indices = np.linspace(0, point_indices.size - 1, num=int(max_points_per_frame), dtype=np.int64)
        source_world: dict[int, np.ndarray] = {}
        source_valid: dict[int, bool] = {}
        for q in point_indices:
            sf = int(src_frame_global[q])
            depth = stream.load_depth(sf)
            h, w = depth.shape[:2]
            x = int(src_xy[q, 0])
            y = int(src_xy[q, 1])
            if x < 0 or x >= w or y < 0 or y >= h:
                source_valid[int(q)] = False
                continue
            z = float(depth[y, x])
            if not np.isfinite(z) or z <= 1e-6:
                source_valid[int(q)] = False
                continue
            src_cam = _backproject_pixel(float(x), float(y), z, intr)
            pose_src = stream.load_pose(sf).astype(np.float64)
            source_world[int(q)] = (pose_src @ np.asarray([*src_cam, 1.0], dtype=np.float64))[:3]
            source_valid[int(q)] = True
        scene_samples_by_control: dict[str, list[dict[str, Any]]] = {name: [] for name in controls}
        for local_idx, frame_id in enumerate(frame_ids[:num_frames]):
            depth_t = stream.load_depth(int(frame_id))
            h, w = depth_t.shape[:2]
            pose_t_inv = np.linalg.inv(stream.load_pose(int(frame_id)).astype(np.float64))
            frame_samples_by_control: dict[str, list[dict[str, Any]]] = {name: [] for name in controls}
            perm = np.asarray(point_indices)
            if perm.size:
                perm = perm[rng.permutation(perm.size)]
            for pos, q in enumerate(point_indices):
                if not source_valid.get(int(q), False):
                    continue
                world = source_world[int(q)]
                cam = (pose_t_inv @ np.asarray([*world, 1.0], dtype=np.float64))[:3]
                gt_pix = _project_camera(cam, intr)
                gt_in_frame = False
                pseudo_visible = False
                if gt_pix is not None:
                    gx, gy = gt_pix
                    xi = int(round(gx))
                    yi = int(round(gy))
                    gt_in_frame = 0 <= xi < w and 0 <= yi < h
                    if gt_in_frame:
                        gt_z = float(depth_t[yi, xi])
                        pseudo_visible = gt_z > 1e-6 and abs(gt_z - float(cam[2])) <= float(depth_consistency_m)
                pred_vis_base = (
                    bool(valid[local_idx, q])
                    and float(visibility[local_idx, q]) >= float(visibility_threshold)
                    and float(confidence[local_idx, q]) >= 0.5
                )
                preds: dict[str, np.ndarray] = {
                    "real_d4rt": uv_pred[local_idx, q],
                    "shuffle_target_frames": uv_pred[(local_idx + 1) % num_frames, q],
                    "source_uv_no_motion": src_uv[q],
                    "random_same_count": rng.random(2),
                }
                preds["shuffle_source_points"] = uv_pred[local_idx, int(perm[pos])] if perm.size else uv_pred[local_idx, q]
                for control, uv in preds.items():
                    px = float(uv[0]) * float(max(w - 1, 1))
                    py = float(uv[1]) * float(max(h - 1, 1))
                    epe = float("nan")
                    if pseudo_visible and gt_pix is not None:
                        epe = float(math.hypot(px - gt_pix[0], py - gt_pix[1]))
                    pred_visible = pred_vis_base
                    if control == "random_same_count":
                        pred_visible = bool(valid[local_idx, q])
                    sample = {
                        "scene": scene,
                        "frame_id": int(frame_id),
                        "local_idx": int(local_idx),
                        "control": control,
                        "query_index": int(q),
                        "epe": epe,
                        "pseudo_visible": bool(pseudo_visible),
                        "gt_in_frame": bool(gt_in_frame),
                        "pred_visible": bool(pred_visible),
                        "visibility": float(visibility[local_idx, q]),
                        "confidence": float(confidence[local_idx, q]),
                    }
                    frame_samples_by_control[control].append(sample)
                    scene_samples_by_control[control].append(sample)
            for control, samples in frame_samples_by_control.items():
                frame_rows.append(
                    {
                        "scene": scene,
                        "frame_id": int(frame_id),
                        "control": control,
                        **_summarize_track_samples(samples),
                    }
                )
        for control, samples in scene_samples_by_control.items():
            scene_rows.append({"scene": scene, "control": control, **_summarize_track_samples(samples)})
    summary_rows = []
    for control in controls:
        control_rows = [row for row in scene_rows if row.get("control") == control and row.get("status") == "ok"]
        merged: dict[str, Any] = {"control": control, "scene_count": int(len(control_rows))}
        for key in ("EPE", "EPE_median", "EPE_p90", "PCK@1px", "PCK@3px", "PCK@5px", "PCK@10px", "visibility_precision", "visibility_recall", "visibility_F1"):
            values = [_to_float(row.get(key)) for row in control_rows]
            values = [v for v in values if v is not None]
            merged[key] = float(np.mean(values)) if values else None
        summary_rows.append(merged)
    _write_csv(out_dir / "track_frame_rows.csv", frame_rows)
    _write_csv(out_dir / "track_scene_rows.csv", scene_rows)
    _write_csv(out_dir / "track_summary.csv", summary_rows)
    _write_json(out_dir / "track_summary.json", summary_rows)
    _write_json(out_dir / "manifest.json", {**DIAG_MANIFEST, "alignment_source": "none", "phase": "v23_tracks", "pseudo_gt_from_scannet_depth_pose": True})
    return summary_rows


def build_pose_tables(v22_audit_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    source = v22_audit_root / "v22_10_ref0_trajectory_consistency_probe5" / "ref0_trajectory_scene_summary.csv"
    frame_source = v22_audit_root / "v22_10_ref0_trajectory_consistency_probe5" / "ref0_trajectory_frame_rows.csv"
    frame_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for frame_row in _read_csv(frame_source):
        scene = frame_row.get("scene", "")
        for out_key, source_key in [
            ("rotation", "rot_err_ref_to_target_deg"),
            ("translation_direction", "trans_dir_err_ref_to_target_deg"),
        ]:
            value = _to_float(frame_row.get(source_key))
            if scene and value is not None:
                frame_values[scene][out_key].append(value)
    rows = []
    for row in _read_csv(source):
        scene = row.get("scene", "")
        rot_values = frame_values[scene]["rotation"]
        dir_values = frame_values[scene]["translation_direction"]
        rows.append(
            {
                "variant": "C1",
                "label": "rigid xyz_ref0 -> xyz_local per target frame",
                "scene": scene,
                "median_rotation_error_deg": _to_float(row.get("rot_err_ref_to_target_median_deg")),
                "p90_rotation_error_deg": float(np.percentile(rot_values, 90)) if rot_values else None,
                "median_translation_direction_error_deg": _to_float(row.get("trans_dir_err_ref_to_target_median_deg")),
                "p90_translation_direction_error_deg": float(np.percentile(dir_values, 90)) if dir_values else None,
                "translation_scale_ratio_median": _to_float(row.get("ratio_median")),
                "scale_absrel_vs_R23": _to_float(row.get("ratio_median_abs_rel_vs_eval_scale")),
                "rigid_residual_p90_median": _to_float(row.get("rigid_residual_p90_median")),
                "source_artifact": str(source),
                "status": "ok",
            }
        )
    _write_csv(out_dir / "pose_scene_summary.csv", rows)
    _write_json(out_dir / "pose_scene_summary.json", rows)
    _copy_if_exists(v22_audit_root / "v22_10_ref0_trajectory_consistency_probe5" / "ref0_trajectory_frame_rows.csv", out_dir / "pose_frame_rows.csv")
    _write_json(out_dir / "manifest.json", {**DIAG_MANIFEST, "alignment_source": "pose_trajectory", "phase": "v23_pose"})
    return rows


def build_stitching_tables(v22_audit_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    source = v22_audit_root / "v22_phaseB" / "native_geometry_diagnostics_v22_phaseB_phase_c_rows.csv"
    rows = []
    for row in _read_csv(source):
        rows.append(
            {
                "variant": "S2",
                "scene": row.get("scene"),
                "prev_window": row.get("prev_window"),
                "curr_window": row.get("curr_window"),
                "overlap_anchor_count": _to_float(row.get("overlap_anchor_count")),
                "stable_id_match_count": _to_float(row.get("match_source_stable_id_count")),
                "mutual_uv_match_count": _to_float(row.get("match_source_mutual_uv_count")),
                "cycle_consistent_match_ratio": _to_float(row.get("cycle_consistency_pass_ratio")),
                "residual_median": _to_float(row.get("self_sim3_residual_median")),
                "residual_p90": _to_float(row.get("self_sim3_residual_p90")),
                "residual_p95": _to_float(row.get("self_sim3_residual_p95")),
                "inlier_abs005": _to_float(row.get("self_sim3_inlier_ratio_abs005")),
                "inlier_abs010": _to_float(row.get("self_sim3_inlier_ratio_abs010")),
                "inlier_rel001": _to_float(row.get("self_sim3_inlier_ratio_rel001")),
                "inlier_rel002": _to_float(row.get("self_sim3_inlier_ratio_rel002")),
                "scale": _to_float(row.get("self_sim3_scale")),
                "status": "ok" if str(row.get("self_sim3_success")) == "True" else "failed",
                "source_artifact": str(source),
            }
        )
    _write_csv(out_dir / "stitching_pair_rows.csv", rows)
    if rows:
        scales = [row["scale"] for row in rows if row.get("scale") is not None]
        summary = [
            {
                "variant": "S2",
                "pair_count": len(rows),
                "residual_p90_mean": float(np.mean([row["residual_p90"] for row in rows if row.get("residual_p90") is not None])),
                "inlier_abs010_mean": float(np.mean([row["inlier_abs010"] for row in rows if row.get("inlier_abs010") is not None])),
                "scale_std": float(np.std(scales)) if scales else None,
                "accumulated_scale_drift": float(abs(np.prod(scales) - 1.0)) if scales else None,
                "post_stitch_fscore10": None,
                "note": "Post-stitch F-score comes from direct R2/R3 scene0050 rows in pointcloud_summary; no separate v23 rerun.",
            }
        ]
    else:
        summary = [{"variant": "S2", "status": "missing_source_artifact"}]
    _write_csv(out_dir / "stitching_summary.csv", summary)
    _write_json(out_dir / "stitching_summary.json", summary)
    _write_json(out_dir / "manifest.json", {**DIAG_MANIFEST, "alignment_source": "self_sim3", "phase": "v23_stitching"})
    return summary


def build_scale_tables(v22_audit_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(v22_audit_root / "v22_12_ref0_scale_convention_probe5" / "ref0_scale_convention_candidate_errors.csv"):
        rows.append(
            {
                "source": "v22_12_scale_convention",
                "predictor": row.get("candidate"),
                "AbsRel_mean": _to_float(row.get("mean_abs_rel_vs_eval_scale")),
                "AbsRel_median": _to_float(row.get("median_abs_rel_vs_eval_scale")),
                "AbsRel_max": _to_float(row.get("max_abs_rel_vs_eval_scale")),
                "mean_candidate_scale": _to_float(row.get("mean_candidate_scale")),
                "uses_scannet_depth_for_metric": True,
            }
        )
    for row in _read_csv(v22_audit_root / "v22_16_target_scale_observability_probe5" / "target_scale_observability_predictor_summary.csv"):
        rows.append(
            {
                "source": "v22_16_observability",
                "predictor": row.get("predictor"),
                "label": row.get("label"),
                "AbsRel_mean": _to_float(row.get("mean_absrel")),
                "AbsRel_median": _to_float(row.get("median_absrel")),
                "AbsRel_max": _to_float(row.get("max_absrel")),
                "uses_pose_feature": row.get("uses_pose_feature"),
                "uses_scannet_depth_for_metric": True,
            }
        )
    for row in _read_csv(v22_audit_root / "v22_16_target_scale_observability_probe5" / "target_scale_observability_univariate_summary.csv"):
        rows.append(
            {
                "source": "v22_16_univariate_observability",
                "predictor": row.get("predictor"),
                "label": row.get("label_key"),
                "feature": row.get("feature_key"),
                "AbsRel_mean": _to_float(row.get("mean_absrel")),
                "AbsRel_median": _to_float(row.get("median_absrel")),
                "AbsRel_max": _to_float(row.get("max_absrel")),
                "uses_scannet_depth_for_metric": True,
                "uses_scannet_pose_for_features": row.get("uses_scannet_pose_for_features"),
            }
        )
    tolerance = _read_csv(v22_audit_root / "v22_19_scale_anchor_tolerance_probe5" / "scale_anchor_tolerance_summary.csv")
    if tolerance:
        for row in tolerance:
            rows.append(
                {
                    "source": "v22_19_tolerance",
                    "predictor": f"R23_oracle_scale_x{row.get('scale_multiplier')}",
                    "relative_scale_error": _to_float(row.get("relative_scale_error")),
                    "F10_retention": _to_float(row.get("fscore@10cm_retention_vs_oracle")),
                    "F10_mean": _to_float(row.get("fscore@10cm_mean")),
                    "depth_delta1_mean": _to_float(row.get("depth_raw_delta1_mean")),
                    "uses_scannet_depth_for_metric": True,
                }
            )
    _write_csv(out_dir / "scale_summary.csv", rows)
    _write_json(out_dir / "scale_summary.json", rows)
    _write_json(out_dir / "manifest.json", {**DIAG_MANIFEST, "alignment_source": "mixed", "phase": "v23_scale"})
    return rows


def build_branch_audit(v22_audit_root: Path, cache_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = _read_csv(v22_audit_root / "v22_3_local_vs_ref_probe5" / "local_vs_ref_probe5.csv")
    summary: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        branch = row.get("branch", "")
        for key in ("target_reproj_error_px_median", "target_reproj_error_px_p90", "target_positive_z_rate", "source_self_uv_error_px_median"):
            value = _to_float(row.get(key))
            if value is not None:
                summary[f"{branch}_{key}"].append(value)
    out_rows = [{"metric": key, "mean": float(np.mean(values)), "count": len(values)} for key, values in sorted(summary.items())]
    key_rows = []
    for path in sorted(cache_root.glob("*/carriers_window*.npz")):
        with np.load(path) as data:
            keys = set(data.files)
            key_rows.append(
                {
                    "cache_file": str(path),
                    "has_xyz_local": "xyz_local" in keys,
                    "has_xyz_ref0": "xyz_ref" in keys,
                    "has_uv_pred": "uv_pred" in keys,
                    "has_visibility": "visibility_prob" in keys,
                    "has_confidence": "confidence_prob" in keys,
                }
            )
    _write_csv(out_dir / "branch_summary.csv", out_rows)
    _write_csv(out_dir / "cache_key_audit.csv", key_rows)
    _write_json(out_dir / "manifest.json", {**DIAG_MANIFEST, "alignment_source": "none", "phase": "v23_branch_audit"})
    return out_rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_integrated_report(
    report_path: Path,
    *,
    branch_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    pose_rows: list[dict[str, Any]],
    track_rows: list[dict[str, Any]],
    stitch_rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
) -> None:
    best_eval = next((row for row in point_rows if row.get("variant") == "P5"), {})
    raw = next((row for row in point_rows if row.get("variant") == "P0"), {})
    scale_candidates = [
        row
        for row in scale_rows
        if row.get("AbsRel_mean") is not None and "oracle" not in str(row.get("predictor", "")).lower()
    ]
    best_scale = min(
        scale_candidates,
        key=lambda r: float(r["AbsRel_mean"]),
        default={},
    )
    real_track = next((row for row in track_rows if row.get("control") == "real_d4rt"), {})
    shuffle_track = next((row for row in track_rows if row.get("control") == "shuffle_target_frames"), {})
    lines = [
        "# Stream4D v23 D4RT Reconstruction Quality Report",
        "",
        "Diagnostic-only. This report uses ScanNet depth/pose/instance labels only as evaluation labels or explicit eval-only alignment sources. It is not a method result.",
        "",
        "## Table 1: Depth",
        _markdown_table(depth_rows, ["variant", "source_variant", "AbsRel", "RMSE", "delta1", "delta2", "delta3", "valid_pixel_ratio", "linear_scale", "linear_shift", "alignment_source"]),
        "",
        "## Table 2: Point Cloud",
        _markdown_table(point_rows, ["variant", "source_variant", "scope", "F@5cm", "F@10cm", "F@20cm", "Chamfer_L1", "Precision@10cm", "Recall@20cm", "Outlier@20cm", "alignment_source"]),
        "",
        "## Table 3: Camera Pose",
        _markdown_table(pose_rows, ["variant", "scene", "median_rotation_error_deg", "p90_rotation_error_deg", "median_translation_direction_error_deg", "translation_scale_ratio_median", "scale_absrel_vs_R23"]),
        "",
        "## Table 4: Track",
        _markdown_table(track_rows, ["control", "scene_count", "EPE", "PCK@1px", "PCK@3px", "PCK@5px", "PCK@10px", "visibility_F1"]),
        "",
        "## Table 5: Chunk Stitching",
        _markdown_table(stitch_rows, ["variant", "pair_count", "residual_p90_mean", "inlier_abs010_mean", "scale_std", "accumulated_scale_drift", "post_stitch_fscore10"]),
        "",
        "## Table 6: Scale Anchor",
        _markdown_table(scale_rows[:20], ["source", "predictor", "AbsRel_mean", "AbsRel_median", "AbsRel_max", "F10_retention"]),
        "",
        "## Main Conclusions",
        f"- Raw camera-space geometry P0 F@10 = {raw.get('F@10cm', 'NA')}.",
        f"- Eval ref0 pose + scale P5 F@10 = {best_eval.get('F@10cm', 'NA')}.",
        f"- Best scale diagnostic row = {best_scale.get('predictor', 'NA')} with mean AbsRel = {best_scale.get('AbsRel_mean', 'NA')}.",
        f"- Real D4RT track PCK@10px = {real_track.get('PCK@10px', 'NA')}; shuffle-target PCK@10px = {shuffle_track.get('PCK@10px', 'NA')}.",
        "- Dominant blocker remains metric scale / canonical anchor if raw/self F@10 is very low while eval-aligned F@10 is high.",
        "- Direct semantic 4D method table is not generated in v23.",
        "",
        "## Scene Categories",
        "- scene0030_00: hard scene for trajectory scale drift and scale proxy stability.",
        "- scene0081_01: hard scene for ref0 scale residual / lower eval upper-bound.",
        "- Other probe5 scenes: mostly good relative geometry under eval-only alignment but poor raw metric geometry.",
        "",
        "## Artifact Roots",
        "- Stream3D/outputs/audit/v23_depth",
        "- Stream3D/outputs/audit/v23_pointcloud",
        "- Stream3D/outputs/audit/v23_pose",
        "- Stream3D/outputs/audit/v23_tracks",
        "- Stream3D/outputs/audit/v23_stitching",
        "- Stream3D/outputs/audit/v23_scale",
        "- Stream3D/outputs/audit/v23_integrated_report",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_code_packet(repo_root: Path, packet_root: Path) -> dict[str, Any]:
    filelist_path = repo_root / "stream4d_v23_filelist.txt"
    zip_path = repo_root / "stream4d_v23_code_review_packet.zip"
    sha_path = repo_root / "stream4d_v23_code_review_packet.sha256"
    ziptest_path = repo_root / "stream4d_v23_ziptest.log"
    include_roots = [
        "docs/stream4d_v23_d4rt_reconstruction_quality_plan.md",
        "docs/stream4d_v23_d4rt_reconstruction_quality_report.md",
        "docs/stream4d_v23_执行日志.md",
        "docs/stream4d_v23_实验结果复盘.md",
        "Stream3D/tools/run_v23_d4rt_reconstruction_quality_audit.py",
        "Stream3D/tests/test_v23_d4rt_reconstruction_quality.py",
    ]
    for pattern_root in [
        "Stream3D/tools/run_v22_direct_reconstruction_benchmark.py",
        "Stream3D/tools/diagnose_v22_d4rt_local_vs_ref.py",
        "Stream3D/tools/diagnose_v22_ref0_trajectory_scale.py",
        "Stream3D/tools/diagnose_v22_ref0_trajectory_policy_sweep.py",
        "Stream3D/tools/diagnose_v22_ref0_scale_convention.py",
        "Stream3D/tools/diagnose_v22_target_scale_observability.py",
        "Stream3D/tools/diagnose_v22_scale_anchor_tolerance.py",
        "Stream3D/stream4d/d4rt_adapter.py",
        "Stream3D/stream4d/carrier_store.py",
        "Stream3D/stream4d_native/self_stitch.py",
        "Stream3D/stream4d_native/sim3.py",
        "Stream3D/geometry_provider/d4rt_carrier_provider.py",
        "Stream3D/tests/test_v22_direct_reconstruction.py",
    ]:
        include_roots.append(pattern_root)
    include_roots.extend(str(path) for path in sorted((repo_root / "Stream3D/outputs/audit").glob("v23_*/*.csv")))
    include_roots.extend(str(path) for path in sorted((repo_root / "Stream3D/outputs/audit").glob("v23_*/*.json")))
    include_roots.extend(str(path) for path in sorted((repo_root / "Stream3D/outputs/audit").glob("v23_*/*.md")))
    rel_files = []
    for raw in include_roots:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        if path.exists() and path.is_file():
            rel_files.append(str(path.relative_to(repo_root)))
    rel_files = sorted(set(rel_files))
    filelist_path.write_text("\n".join(rel_files) + "\n", encoding="utf-8")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in rel_files:
            zf.write(repo_root / rel, arcname=rel)
    import hashlib

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sha_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    ziptest_path.write_text("OK\n" if bad is None else f"BAD {bad}\n", encoding="utf-8")
    return {
        "filelist": str(filelist_path),
        "zip": str(zip_path),
        "sha256": str(sha_path),
        "ziptest": str(ziptest_path),
        "file_count": len(rel_files),
        "zip_test_ok": bad is None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stream4D v23 D4RT reconstruction quality audit artifacts.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--audit-root", default="outputs/audit")
    parser.add_argument("--v22-audit-root", default="outputs/audit")
    parser.add_argument("--local-cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-track-points-per-frame", type=int, default=512)
    parser.add_argument("--track-depth-consistency-m", type=float, default=0.10)
    parser.add_argument("--track-visibility-threshold", type=float, default=0.5)
    parser.add_argument("--report-path", default="../docs/stream4d_v23_d4rt_reconstruction_quality_report.md")
    parser.add_argument("--make-code-packet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)
    v22_audit_root = Path(args.v22_audit_root)
    local_cache_root = Path(args.local_cache_root)
    seq_list = _read_seq_list(Path(args.seq_list))
    branch_rows = build_branch_audit(v22_audit_root, local_cache_root, audit_root / "v23_integrated_report")
    depth_rows = build_depth_tables(v22_audit_root, audit_root / "v23_depth")
    point_rows = build_pointcloud_tables(v22_audit_root, audit_root / "v23_pointcloud")
    pose_rows = build_pose_tables(v22_audit_root, audit_root / "v23_pose")
    track_rows = build_track_tables(
        seq_list,
        local_cache_root,
        audit_root / "v23_tracks",
        backbone=args.backbone,
        max_points_per_frame=int(args.max_track_points_per_frame),
        depth_consistency_m=float(args.track_depth_consistency_m),
        visibility_threshold=float(args.track_visibility_threshold),
    )
    stitch_rows = build_stitching_tables(v22_audit_root, audit_root / "v23_stitching")
    scale_rows = build_scale_tables(v22_audit_root, audit_root / "v23_scale")
    integrated_dir = audit_root / "v23_integrated_report"
    integrated_payload = {
        "branch": branch_rows,
        "depth": depth_rows,
        "pointcloud": point_rows,
        "pose": pose_rows,
        "tracks": track_rows,
        "stitching": stitch_rows,
        "scale": scale_rows,
        "manifest": {**DIAG_MANIFEST, "phase": "v23_integrated_report", "alignment_source": "mixed"},
    }
    _write_json(integrated_dir / "integrated_summary.json", integrated_payload)
    _write_json(integrated_dir / "manifest.json", integrated_payload["manifest"])
    build_integrated_report(
        Path(args.report_path),
        branch_rows=branch_rows,
        depth_rows=depth_rows,
        point_rows=point_rows,
        pose_rows=pose_rows,
        track_rows=track_rows,
        stitch_rows=stitch_rows,
        scale_rows=scale_rows,
    )
    code_packet = None
    if args.make_code_packet:
        code_packet = build_code_packet(Path.cwd().parent.resolve(), integrated_dir)
        _write_json(integrated_dir / "code_packet_metadata.json", code_packet)
    print(
        json.dumps(
            _json_safe(
                {
                    "status": "ok",
                    "seq_count": len(seq_list),
                    "depth_rows": len(depth_rows),
                    "point_rows": len(point_rows),
                    "pose_rows": len(pose_rows),
                    "track_rows": len(track_rows),
                    "stitch_rows": len(stitch_rows),
                    "scale_rows": len(scale_rows),
                    "report_path": str(Path(args.report_path)),
                    "code_packet": code_packet,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
