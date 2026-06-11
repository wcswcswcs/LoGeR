from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

STREAM3D_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider
from stream4d.scannet_stream import ScanNetStream
from tools.diagnose_v22_ref0_trajectory_scale import (
    _finite_values,
    _relative_ref_to_target,
    _rotation_error_deg,
    _translation_direction_error_deg,
)
from tools.run_v22_direct_reconstruction_benchmark import (
    _fit_ref0_pose_scale,
    _fit_rigid_no_scale,
    _json_safe,
    _read_seq_list,
    _sample_indices,
)


_CANDIDATE_KEYS = [
    "trajectory_scale_ratio",
    "target_depth_over_local_z_median",
    "target_depth_over_ref_z_median",
    "source_depth_over_ref_z_median",
    "source_depth_over_local_z_median",
    "target_depth_over_local_z_mean",
    "source_depth_over_ref_z_mean",
]


def _median_positive(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 1e-8)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _mean_positive(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 1e-8)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(float(numerator)) or not np.isfinite(float(denominator)) or float(denominator) <= 1e-8:
        return None
    return float(float(numerator) / float(denominator))


def _sample_depth_xy(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = arr.shape[:2]
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.full((points.shape[0],), np.nan, dtype=np.float32)
    out[ok] = arr[y[ok], x[ok]]
    return out


def _sample_depth_uv(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    points = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = arr.shape[:2]
    xy = np.stack([points[:, 0] * float(max(w - 1, 1)), points[:, 1] * float(max(h - 1, 1))], axis=1)
    return _sample_depth_xy(arr, xy)


def _load_depth_cached(stream: ScanNetStream, cache: dict[int, np.ndarray], frame_id: int) -> np.ndarray:
    key = int(frame_id)
    if key not in cache:
        cache[key] = stream.load_depth(key)
    return cache[key]


def _source_depths(
    stream: ScanNetStream,
    depth_cache: dict[int, np.ndarray],
    src_frame_global: np.ndarray,
    src_xy: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    out = np.full((indices.shape[0],), np.nan, dtype=np.float32)
    frames = np.asarray(src_frame_global, dtype=np.int64).reshape(-1)
    xy_all = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    for frame_id in np.unique(frames[indices]):
        local = np.flatnonzero(frames[indices] == int(frame_id))
        if local.size == 0:
            continue
        depth = _load_depth_cached(stream, depth_cache, int(frame_id))
        out[local] = _sample_depth_xy(depth, xy_all[indices[local]])
    return out


def _candidate_error(value: float | None, eval_scale: float | None) -> float | None:
    if value is None or eval_scale is None:
        return None
    if not np.isfinite(float(value)) or not np.isfinite(float(eval_scale)) or float(eval_scale) <= 1e-8:
        return None
    return float(abs(float(value) - float(eval_scale)) / float(eval_scale))


def _diagnose_window(
    stream: ScanNetStream,
    window: Any,
    *,
    scene: str,
    window_index: int,
    max_anchors: int,
    robust_trim_percentile: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not window.frame_ids:
        return [], {"status": "no_frames"}
    pose0 = stream.load_pose(int(window.frame_ids[0]))
    if not np.isfinite(pose0).all():
        return [], {"status": "invalid_pose0"}
    _, eval_diag = _fit_ref0_pose_scale(
        stream,
        window,
        robust_trim_percentile=float(robust_trim_percentile),
        max_anchors=int(max_anchors),
    )
    eval_scale = eval_diag.get("ref0_pose_scale")
    with np.load(window.path) as data:
        if "xyz_local" not in data.files:
            return [], {"status": "missing_xyz_local", **eval_diag}
        xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float64)
        xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
        src_frame = np.asarray(data.get("src_frame", np.zeros((xyz_ref.shape[1],), dtype=np.int64)), dtype=np.int64)
        src_frame_global = np.asarray(data.get("src_frame_global", src_frame), dtype=np.int64)
        src_xy = np.asarray(data.get("src_xy", np.full((xyz_ref.shape[1], 2), np.nan)), dtype=np.float64)
    if xyz_local.shape != np.asarray(window.xyz).shape or xyz_ref.shape != xyz_local.shape:
        return [], {"status": "shape_mismatch", **eval_diag}

    frame_rows: list[dict[str, Any]] = []
    depth_cache: dict[int, np.ndarray] = {}
    per_frame_cap = max(4, int(max_anchors) // max(len(window.frame_ids) - 1, 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        if local_idx == 0:
            continue
        pose = stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            continue
        uv = np.asarray(window.uv[local_idx], dtype=np.float64)
        ok = (
            np.asarray(window.valid[local_idx], dtype=bool)
            & np.isfinite(xyz_ref[local_idx]).all(axis=1)
            & np.isfinite(xyz_local[local_idx]).all(axis=1)
            & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= 0.5)
            & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= 0.5)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
        )
        indices = np.flatnonzero(ok)
        if indices.shape[0] < 4:
            continue
        indices = _sample_indices(indices, per_frame_cap)
        try:
            rot, trans, residual = _fit_rigid_no_scale(
                xyz_ref[local_idx, indices],
                xyz_local[local_idx, indices],
            )
        except Exception:
            continue

        rel_ref_to_target = _relative_ref_to_target(pose0, pose)
        d4rt_len = float(np.linalg.norm(trans))
        pose_len = float(np.linalg.norm(rel_ref_to_target[:3, 3]))
        ratio = float(pose_len / d4rt_len) if d4rt_len > 1e-8 and pose_len > 1e-8 else None

        target_depth = _sample_depth_uv(_load_depth_cached(stream, depth_cache, int(frame_id)), uv[indices])
        source_depth = _source_depths(stream, depth_cache, src_frame_global, src_xy, indices)
        ref_z_abs = np.abs(xyz_ref[local_idx, indices, 2])
        local_z_abs = np.abs(xyz_local[local_idx, indices, 2])
        ref_norm = np.linalg.norm(xyz_ref[local_idx, indices], axis=1)
        local_norm = np.linalg.norm(xyz_local[local_idx, indices], axis=1)

        target_depth_median = _median_positive(target_depth)
        target_depth_mean = _mean_positive(target_depth)
        source_depth_median = _median_positive(source_depth)
        source_depth_mean = _mean_positive(source_depth)
        ref_z_median = _median_positive(ref_z_abs)
        ref_z_mean = _mean_positive(ref_z_abs)
        local_z_median = _median_positive(local_z_abs)
        local_z_mean = _mean_positive(local_z_abs)
        ref_norm_median = _median_positive(ref_norm)
        local_norm_median = _median_positive(local_norm)

        row: dict[str, Any] = {
            "scene": scene,
            "window_index": int(window_index),
            "window_path": str(window.path),
            "ref_frame": int(window.frame_ids[0]),
            "frame_id": int(frame_id),
            "local_idx": int(local_idx),
            "anchor_count": int(indices.shape[0]),
            "target_depth_valid": int(np.count_nonzero(np.isfinite(target_depth) & (target_depth > 0.0))),
            "source_depth_valid": int(np.count_nonzero(np.isfinite(source_depth) & (source_depth > 0.0))),
            "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
            "d4rt_translation_norm": d4rt_len,
            "pose_translation_norm": pose_len,
            "trajectory_scale_ratio": ratio,
            "rot_err_ref_to_target_deg": _rotation_error_deg(rot, rel_ref_to_target[:3, :3]),
            "trans_dir_err_ref_to_target_deg": _translation_direction_error_deg(trans, rel_ref_to_target[:3, 3]),
            "rigid_residual_median": float(np.median(residual)),
            "rigid_residual_p90": float(np.percentile(residual, 90)),
            "target_depth_median": target_depth_median,
            "target_depth_mean": target_depth_mean,
            "source_depth_median": source_depth_median,
            "source_depth_mean": source_depth_mean,
            "pred_ref_abs_z_median": ref_z_median,
            "pred_ref_abs_z_mean": ref_z_mean,
            "pred_local_abs_z_median": local_z_median,
            "pred_local_abs_z_mean": local_z_mean,
            "pred_ref_norm_median": ref_norm_median,
            "pred_local_norm_median": local_norm_median,
            "local_over_ref_z_median": _safe_ratio(local_z_median, ref_z_median),
            "local_over_ref_norm_median": _safe_ratio(local_norm_median, ref_norm_median),
            "target_over_source_depth_median": _safe_ratio(target_depth_median, source_depth_median),
            "target_depth_over_local_z_median": _safe_ratio(target_depth_median, local_z_median),
            "target_depth_over_ref_z_median": _safe_ratio(target_depth_median, ref_z_median),
            "source_depth_over_ref_z_median": _safe_ratio(source_depth_median, ref_z_median),
            "source_depth_over_local_z_median": _safe_ratio(source_depth_median, local_z_median),
            "target_depth_over_local_z_mean": _safe_ratio(target_depth_mean, local_z_mean),
            "source_depth_over_ref_z_mean": _safe_ratio(source_depth_mean, ref_z_mean),
        }
        for key in _CANDIDATE_KEYS:
            row[f"{key}_abs_rel_vs_eval_scale"] = _candidate_error(row.get(key), row.get("eval_ref0_depth_scale"))
        frame_rows.append(row)
    return frame_rows, {"status": "ok", **eval_diag}


def _scene_summary(scene: str, window_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eval_values = _finite_values(window_rows, "eval_ref0_depth_scale")
    eval_scale = float(np.median(eval_values)) if eval_values.size else None
    summary: dict[str, Any] = {
        "scene": scene,
        "status": "ok" if frame_rows else "no_frame_rows",
        "window_count": int(len(window_rows)),
        "frame_count": int(len(frame_rows)),
        "eval_ref0_depth_scale": eval_scale,
    }
    for key in [
        "d4rt_translation_norm",
        "pose_translation_norm",
        "rigid_residual_p90",
        "rot_err_ref_to_target_deg",
        "trans_dir_err_ref_to_target_deg",
        "target_depth_median",
        "source_depth_median",
        "pred_ref_abs_z_median",
        "pred_local_abs_z_median",
        "local_over_ref_z_median",
        "target_over_source_depth_median",
        *_CANDIDATE_KEYS,
    ]:
        values = _finite_values(frame_rows, key)
        summary[f"{key}_median"] = float(np.median(values)) if values.size else None
        summary[f"{key}_mean"] = float(np.mean(values)) if values.size else None
    for key in _CANDIDATE_KEYS:
        value = summary.get(f"{key}_median")
        summary[f"{key}_median_abs_rel_vs_eval_scale"] = _candidate_error(value, eval_scale)
    return summary


def _candidate_error_rows(scene_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in _CANDIDATE_KEYS:
        errors = _finite_values(scene_summaries, f"{key}_median_abs_rel_vs_eval_scale")
        values = _finite_values(scene_summaries, f"{key}_median")
        if errors.size == 0:
            continue
        rows.append(
            {
                "candidate": key,
                "scene_count": int(errors.size),
                "mean_abs_rel_vs_eval_scale": float(np.mean(errors)),
                "median_abs_rel_vs_eval_scale": float(np.median(errors)),
                "max_abs_rel_vs_eval_scale": float(np.max(errors)),
                "mean_candidate_scale": float(np.mean(values)) if values.size else None,
            }
        )
    rows.sort(key=lambda row: (float(row["mean_abs_rel_vs_eval_scale"]), float(row["max_abs_rel_vs_eval_scale"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, scene_summaries: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return "NA"
            return f"{float(value):.6f}"
        return str(value)

    lines: list[str] = [
        "# v22.12 ref0 scale-convention diagnostic",
        "",
        "Diagnostic-only: compares D4RT predicted z/depth scale clues against R23 eval-only ref0 depth scale. This does not define a method result.",
        "",
        "## Candidate Error",
        "",
        "| candidate | mean absrel vs eval scale | median absrel | max absrel | mean candidate | scenes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("candidate")),
                    fmt(row.get("mean_abs_rel_vs_eval_scale")),
                    fmt(row.get("median_abs_rel_vs_eval_scale")),
                    fmt(row.get("max_abs_rel_vs_eval_scale")),
                    fmt(row.get("mean_candidate_scale")),
                    fmt(row.get("scene_count")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-scene",
            "",
            "| scene | eval scale | traj ratio | target/local z | source/ref z | target/ref z | source/local z | local/ref z | target/source depth | frames |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scene_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("scene")),
                    fmt(row.get("eval_ref0_depth_scale")),
                    fmt(row.get("trajectory_scale_ratio_median")),
                    fmt(row.get("target_depth_over_local_z_median_median")),
                    fmt(row.get("source_depth_over_ref_z_median_median")),
                    fmt(row.get("target_depth_over_ref_z_median_median")),
                    fmt(row.get("source_depth_over_local_z_median_median")),
                    fmt(row.get("local_over_ref_z_median_median")),
                    fmt(row.get("target_over_source_depth_median_median")),
                    fmt(row.get("frame_count")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v22 ref0 scale convention and mean-depth scale clues.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v22_12_ref0_scale_convention_probe5")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
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
        max_anchors=int(args.max_anchors),
    )
    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = Path(args.cache_root) / scene
        if not scene_dir.exists():
            scene_summaries.append({"scene": scene, "status": "missing_cache"})
            continue
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        cache = provider._load_scene(scene)
        windows = list(cache["windows"])
        if args.max_windows_per_scene is not None:
            windows = windows[: int(args.max_windows_per_scene)]
        scene_frame_rows: list[dict[str, Any]] = []
        scene_window_rows: list[dict[str, Any]] = []
        for window_index, window in enumerate(windows):
            frame_rows, window_diag = _diagnose_window(
                stream,
                window,
                scene=scene,
                window_index=window_index,
                max_anchors=int(args.max_anchors),
                robust_trim_percentile=float(args.robust_trim_percentile),
            )
            eval_scale = window_diag.get("ref0_pose_scale")
            window_row = {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "ref_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                "window_frame_count": int(len(window.frame_ids)),
                "frame_row_count": int(len(frame_rows)),
                "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                **window_diag,
            }
            scene_window_rows.append(window_row)
            scene_frame_rows.extend(frame_rows)
        all_window_rows.extend(scene_window_rows)
        all_frame_rows.extend(scene_frame_rows)
        scene_summaries.append(_scene_summary(scene, scene_window_rows, scene_frame_rows))

    candidate_rows = _candidate_error_rows(scene_summaries)
    _write_csv(audit_root / "ref0_scale_convention_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "ref0_scale_convention_window_rows.csv", all_window_rows)
    _write_csv(audit_root / "ref0_scale_convention_scene_summary.csv", scene_summaries)
    _write_csv(audit_root / "ref0_scale_convention_candidate_errors.csv", candidate_rows)
    (audit_root / "ref0_scale_convention_frame_rows.json").write_text(json.dumps(_json_safe(all_frame_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_scale_convention_window_rows.json").write_text(json.dumps(_json_safe(all_window_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_scale_convention_scene_summary.json").write_text(json.dumps(_json_safe(scene_summaries), indent=2), encoding="utf-8")
    (audit_root / "ref0_scale_convention_candidate_errors.json").write_text(json.dumps(_json_safe(candidate_rows), indent=2), encoding="utf-8")
    _write_md(audit_root / "ref0_scale_convention.md", scene_summaries, candidate_rows)
    print(f"Wrote v22.12 ref0 scale-convention diagnostic to {audit_root}")


if __name__ == "__main__":
    main()
