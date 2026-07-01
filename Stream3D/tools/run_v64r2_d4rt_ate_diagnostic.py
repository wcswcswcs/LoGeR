from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

STREAM3D_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.sim3 import fit_sim3_umeyama
from tools.run_v22_direct_reconstruction_benchmark import (
    _fit_rigid_no_scale,
    _json_safe,
    _read_seq_list,
    _sample_indices,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _norm_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "ate_frame_count": 0,
            "ate_sim3_rmse_m": None,
            "ate_sim3_median_m": None,
            "ate_sim3_p90_m": None,
            "ate_sim3_max_m": None,
        }
    return {
        "ate_frame_count": int(values.size),
        "ate_sim3_rmse_m": float(math.sqrt(float(np.mean(values**2)))),
        "ate_sim3_median_m": float(np.median(values)),
        "ate_sim3_p90_m": float(np.percentile(values, 90)),
        "ate_sim3_max_m": float(np.max(values)),
    }


def _fit_ate(pred_centers: np.ndarray, gt_centers: np.ndarray) -> tuple[dict[str, Any] | None, np.ndarray]:
    pred_centers = np.asarray(pred_centers, dtype=np.float64)
    gt_centers = np.asarray(gt_centers, dtype=np.float64)
    ok = np.isfinite(pred_centers).all(axis=1) & np.isfinite(gt_centers).all(axis=1)
    pred_centers = pred_centers[ok]
    gt_centers = gt_centers[ok]
    if pred_centers.shape[0] < 4:
        return None, np.empty((0,), dtype=np.float64)
    fit = fit_sim3_umeyama(pred_centers, gt_centers)
    residual = np.asarray(fit["residual"], dtype=np.float64)
    return fit, residual


def _window_centers(
    stream: ScanNetStream,
    window: Any,
    *,
    scene: str,
    window_index: int,
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(window.path) as data:
        if "xyz_local" not in data.files:
            return [], np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)

    xyz_ref = np.asarray(window.xyz, dtype=np.float64)
    if xyz_local.shape != xyz_ref.shape:
        return [], np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    pred_centers: list[np.ndarray] = []
    trans_centers: list[np.ndarray] = []
    gt_centers: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    per_frame_cap = max(4, int(max_anchors) // max(len(window.frame_ids) - 1, 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        pose = stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            continue
        gt_center = np.asarray(pose[:3, 3], dtype=np.float64)
        if local_idx == 0:
            pred_center = np.zeros((3,), dtype=np.float64)
            trans_center = np.zeros((3,), dtype=np.float64)
            residual = np.empty((0,), dtype=np.float64)
            anchor_count = 0
        else:
            uv = np.asarray(window.uv[local_idx], dtype=np.float64)
            ok = (
                np.asarray(window.valid[local_idx], dtype=bool)
                & np.isfinite(xyz_ref[local_idx]).all(axis=1)
                & np.isfinite(xyz_local[local_idx]).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= float(min_visibility))
                & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= float(min_confidence))
            )
            indices = np.flatnonzero(ok)
            if indices.shape[0] < 4:
                continue
            indices = _sample_indices(indices, per_frame_cap)
            try:
                rot, trans, residual = _fit_rigid_no_scale(xyz_ref[local_idx, indices], xyz_local[local_idx, indices])
            except Exception:
                continue
            pred_center = -rot.T @ trans
            trans_center = trans.astype(np.float64)
            anchor_count = int(indices.shape[0])

        pred_centers.append(pred_center)
        trans_centers.append(trans_center)
        gt_centers.append(gt_center)
        frame_rows.append(
            {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "frame_id": int(frame_id),
                "local_idx": int(local_idx),
                "anchor_count": int(anchor_count),
                "pred_center_x": float(pred_center[0]),
                "pred_center_y": float(pred_center[1]),
                "pred_center_z": float(pred_center[2]),
                "trans_vector_x": float(trans_center[0]),
                "trans_vector_y": float(trans_center[1]),
                "trans_vector_z": float(trans_center[2]),
                "gt_center_x": float(gt_center[0]),
                "gt_center_y": float(gt_center[1]),
                "gt_center_z": float(gt_center[2]),
                "rigid_residual_median": float(np.median(residual)) if residual.size else None,
                "rigid_residual_p90": float(np.percentile(residual, 90)) if residual.size else None,
            }
        )

    return (
        frame_rows,
        np.asarray(pred_centers, dtype=np.float64),
        np.asarray(trans_centers, dtype=np.float64),
        np.asarray(gt_centers, dtype=np.float64),
    )


def _summarize_window(
    frame_rows: list[dict[str, Any]],
    pred_centers: np.ndarray,
    trans_centers: np.ndarray,
    gt_centers: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not frame_rows:
        return {"status": "no_frame_rows"}, []
    fit, residual = _fit_ate(pred_centers, gt_centers)
    trans_fit, trans_residual = _fit_ate(trans_centers, gt_centers)
    if fit is None:
        return {"status": "too_few_frames", "ate_frame_count": int(len(frame_rows))}, frame_rows

    aligned_rows: list[dict[str, Any]] = []
    for row, value in zip(frame_rows, residual):
        out = dict(row)
        out["ate_sim3_residual_m"] = float(value)
        aligned_rows.append(out)
    summary = {
        "status": "ok",
        "sim3_scale": float(fit["scale"]),
        "sim3_rotation_det": float(fit["rotation_det"]),
        "sim3_anchor_count": int(fit["anchor_count"]),
        **_norm_stats(residual),
    }
    trans_stats = _norm_stats(trans_residual)
    summary.update(
        {
            "trans_vector_ate_sim3_rmse_m": trans_stats["ate_sim3_rmse_m"],
            "trans_vector_ate_sim3_median_m": trans_stats["ate_sim3_median_m"],
            "trans_vector_ate_sim3_p90_m": trans_stats["ate_sim3_p90_m"],
        }
    )
    return summary, aligned_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute diagnostic Sim3-aligned D4RT camera-trajectory ATE.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v64r2_d4rt_ate_probe5")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]

    provider = D4RTCarrierProjectionProvider(
        debug_root=args.cache_root,
        mode="raw",
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        max_anchors=int(args.max_anchors),
    )
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = Path(args.cache_root) / scene
        if not scene_dir.exists():
            scene_rows.append({"scene": scene, "status": "missing_cache"})
            continue
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        cache = provider._load_scene(scene)
        windows = list(cache["windows"])
        if args.max_windows_per_scene is not None:
            windows = windows[: int(args.max_windows_per_scene)]
        scene_residuals: list[float] = []
        scene_window_count = 0
        for window_index, window in enumerate(windows):
            frame_rows, pred_centers, trans_centers, gt_centers = _window_centers(
                stream,
                window,
                scene=scene,
                window_index=window_index,
                max_anchors=int(args.max_anchors),
                min_visibility=float(args.min_visibility),
                min_confidence=float(args.min_confidence),
            )
            summary, aligned_rows = _summarize_window(frame_rows, pred_centers, trans_centers, gt_centers)
            summary.update(
                {
                    "scene": scene,
                    "window_index": int(window_index),
                    "window_path": str(window.path),
                    "frame_count_input": int(len(window.frame_ids)),
                }
            )
            all_window_rows.append(summary)
            all_frame_rows.extend(aligned_rows if aligned_rows else frame_rows)
            for row in aligned_rows:
                value = row.get("ate_sim3_residual_m")
                if value is not None and np.isfinite(float(value)):
                    scene_residuals.append(float(value))
            if summary.get("status") == "ok":
                scene_window_count += 1
        scene_rows.append(
            {
                "scene": scene,
                "status": "ok" if scene_residuals else "no_valid_ate_windows",
                "window_count": int(scene_window_count),
                **_norm_stats(np.asarray(scene_residuals, dtype=np.float64)),
            }
        )

    all_residuals = np.asarray(
        [float(row["ate_sim3_residual_m"]) for row in all_frame_rows if row.get("ate_sim3_residual_m") is not None],
        dtype=np.float64,
    )
    aggregate = {
        "metric": "diagnostic_sim3_aligned_ate",
        "cache_root": str(args.cache_root),
        "seq_list": str(args.seq_list),
        "center_convention": "fit xyz_ref->xyz_local rigid transform, then C_ref=-R^T t",
        "gt_centers": "ScanNet pose translation, used only for evaluation alignment and ATE scoring",
        "method_result": False,
        "notes": "This diagnostic requires xyz_local in the carrier cache; D5 warmstart cache lacks xyz_local, so the default cache is v22_local_xyz_probe5_r1.",
        **_norm_stats(all_residuals),
    }

    _write_csv(audit_root / "d4rt_ate_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "d4rt_ate_window_rows.csv", all_window_rows)
    _write_csv(audit_root / "d4rt_ate_scene_summary.csv", scene_rows)
    (audit_root / "d4rt_ate_summary.json").write_text(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True), encoding="utf-8")
    (audit_root / "d4rt_ate_scene_summary.json").write_text(json.dumps(_json_safe(scene_rows), indent=2, sort_keys=True), encoding="utf-8")
    (audit_root / "d4rt_ate_window_rows.json").write_text(json.dumps(_json_safe(all_window_rows), indent=2, sort_keys=True), encoding="utf-8")
    (audit_root / "d4rt_ate_frame_rows.json").write_text(json.dumps(_json_safe(all_frame_rows), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
