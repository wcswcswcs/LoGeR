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
from tools.diagnose_v22_ref0_trajectory_scale import _finite_values
from tools.run_v22_direct_reconstruction_benchmark import (
    _fit_ref0_pose_scale,
    _json_safe,
    _read_seq_list,
    _sample_indices,
)


_CANDIDATE_KEYS = [
    "scannet_fxy_over_local_fxy",
    "local_fxy_over_scannet_fxy",
    "scannet_fx_over_local_fx",
    "scannet_fy_over_local_fy",
    "local_fx_over_scannet_fx",
    "local_fy_over_scannet_fy",
    "scannet_fxy_over_ref_fxy",
    "ref_fxy_over_scannet_fxy",
    "local_fxy_over_ref_fxy",
    "ref_fxy_over_local_fxy",
]


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 1e-8:
        return None
    return float(num / den)


def _geomean_positive(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    av = float(a)
    bv = float(b)
    if not np.isfinite(av) or not np.isfinite(bv) or av <= 1e-8 or bv <= 1e-8:
        return None
    return float(np.sqrt(av * bv))


def _estimate_intrinsics_params_from_query_geometry(
    pred_tracks: np.ndarray,
    pred_uv_norm: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image_hw
    cx = 0.5 * float(max(width - 1, 1))
    cy = 0.5 * float(max(height - 1, 1))
    uv = np.asarray(pred_uv_norm, dtype=np.float64).reshape(-1, 2)
    pred = np.asarray(pred_tracks, dtype=np.float64).reshape(-1, 3)
    if uv.shape[0] != pred.shape[0]:
        raise ValueError(f"uv/track shape mismatch: {uv.shape[0]} vs {pred.shape[0]}")

    u_px = uv[:, 0] * float(max(width - 1, 1))
    v_px = uv[:, 1] * float(max(height - 1, 1))
    x = pred[:, 0]
    y = pred[:, 1]
    z = pred[:, 2]
    fx_vals = z * np.abs(u_px - cx) / np.maximum(np.abs(x), 1e-6)
    fy_vals = z * np.abs(v_px - cy) / np.maximum(np.abs(y), 1e-6)
    fx_vals = fx_vals[np.isfinite(fx_vals) & (fx_vals > 1e-6)]
    fy_vals = fy_vals[np.isfinite(fy_vals) & (fy_vals > 1e-6)]

    fx = float(np.median(fx_vals)) if fx_vals.size > 0 else float(max(width, 1))
    fy = float(np.median(fy_vals)) if fy_vals.size > 0 else float(max(height, 1))
    params = np.asarray([fx, fy, cx, cy], dtype=np.float64)
    diag = {
        "fx_count": int(fx_vals.size),
        "fy_count": int(fy_vals.size),
        "fx_p10": float(np.percentile(fx_vals, 10)) if fx_vals.size else None,
        "fx_p90": float(np.percentile(fx_vals, 90)) if fx_vals.size else None,
        "fy_p10": float(np.percentile(fy_vals, 10)) if fy_vals.size else None,
        "fy_p90": float(np.percentile(fy_vals, 90)) if fy_vals.size else None,
    }
    return params, diag


def _reprojection_error_px(
    xyz: np.ndarray,
    uv: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_hw: tuple[int, int],
) -> dict[str, Any]:
    height, width = image_hw
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    uv_norm = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & np.isfinite(uv_norm).all(axis=1) & (z > 1e-6)
    if np.count_nonzero(valid) == 0:
        return {"count": 0, "median": None, "p90": None, "mean": None}
    x_proj = points[valid, 0] * float(fx) / z[valid] + float(cx)
    y_proj = points[valid, 1] * float(fy) / z[valid] + float(cy)
    u_px = uv_norm[valid, 0] * float(max(width - 1, 1))
    v_px = uv_norm[valid, 1] * float(max(height - 1, 1))
    err = np.sqrt((x_proj - u_px) ** 2 + (y_proj - v_px) ** 2)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {
        "count": int(err.size),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "mean": float(np.mean(err)),
    }


def _candidate_error(value: float | None, eval_scale: float | None) -> float | None:
    if value is None or eval_scale is None:
        return None
    val = float(value)
    ref = float(eval_scale)
    if not np.isfinite(val) or not np.isfinite(ref) or ref <= 1e-8:
        return None
    return float(abs(val - ref) / ref)


def _image_hw_cached(stream: ScanNetStream, cache: dict[int, tuple[int, int]], frame_id: int) -> tuple[int, int]:
    key = int(frame_id)
    if key not in cache:
        depth = stream.load_depth(key)
        cache[key] = tuple(int(v) for v in depth.shape[:2])
    return cache[key]


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
        xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
        xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float64)
    if xyz_local.shape != np.asarray(window.xyz).shape or xyz_ref.shape != xyz_local.shape:
        return [], {"status": "shape_mismatch", **eval_diag}

    intr = stream.load_intrinsics()
    scannet_fx = float(intr[0, 0])
    scannet_fy = float(intr[1, 1])
    scannet_cx = float(intr[0, 2])
    scannet_cy = float(intr[1, 2])
    scannet_fxy = _geomean_positive(scannet_fx, scannet_fy)

    frame_rows: list[dict[str, Any]] = []
    hw_cache: dict[int, tuple[int, int]] = {}
    per_frame_cap = max(16, int(max_anchors) // max(len(window.frame_ids), 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        uv = np.asarray(window.uv[local_idx], dtype=np.float64)
        ok = (
            np.asarray(window.valid[local_idx], dtype=bool)
            & np.isfinite(xyz_local[local_idx]).all(axis=1)
            & np.isfinite(xyz_ref[local_idx]).all(axis=1)
            & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= 0.5)
            & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= 0.5)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
        )
        indices = np.flatnonzero(ok)
        if indices.shape[0] < 8:
            continue
        indices = _sample_indices(indices, per_frame_cap)
        image_hw = _image_hw_cached(stream, hw_cache, int(frame_id))
        local_params, local_diag = _estimate_intrinsics_params_from_query_geometry(
            xyz_local[local_idx, indices],
            uv[indices],
            image_hw,
        )
        ref_params, ref_diag = _estimate_intrinsics_params_from_query_geometry(
            xyz_ref[local_idx, indices],
            uv[indices],
            image_hw,
        )
        local_fx, local_fy, local_cx, local_cy = [float(v) for v in local_params]
        ref_fx, ref_fy, ref_cx, ref_cy = [float(v) for v in ref_params]
        local_fxy = _geomean_positive(local_fx, local_fy)
        ref_fxy = _geomean_positive(ref_fx, ref_fy)

        local_scan_err = _reprojection_error_px(
            xyz_local[local_idx, indices],
            uv[indices],
            fx=scannet_fx,
            fy=scannet_fy,
            cx=scannet_cx,
            cy=scannet_cy,
            image_hw=image_hw,
        )
        local_pred_err = _reprojection_error_px(
            xyz_local[local_idx, indices],
            uv[indices],
            fx=local_fx,
            fy=local_fy,
            cx=local_cx,
            cy=local_cy,
            image_hw=image_hw,
        )
        ref_scan_err = _reprojection_error_px(
            xyz_ref[local_idx, indices],
            uv[indices],
            fx=scannet_fx,
            fy=scannet_fy,
            cx=scannet_cx,
            cy=scannet_cy,
            image_hw=image_hw,
        )

        row: dict[str, Any] = {
            "scene": scene,
            "window_index": int(window_index),
            "window_path": str(window.path),
            "ref_frame": int(window.frame_ids[0]),
            "frame_id": int(frame_id),
            "local_idx": int(local_idx),
            "anchor_count": int(indices.shape[0]),
            "image_height": int(image_hw[0]),
            "image_width": int(image_hw[1]),
            "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
            "scannet_fx": scannet_fx,
            "scannet_fy": scannet_fy,
            "scannet_fxy": scannet_fxy,
            "local_pred_fx": local_fx,
            "local_pred_fy": local_fy,
            "local_pred_fxy": local_fxy,
            "local_fx_count": int(local_diag["fx_count"]),
            "local_fy_count": int(local_diag["fy_count"]),
            "ref_pred_fx": ref_fx,
            "ref_pred_fy": ref_fy,
            "ref_pred_fxy": ref_fxy,
            "ref_fx_count": int(ref_diag["fx_count"]),
            "ref_fy_count": int(ref_diag["fy_count"]),
            "scannet_fx_over_local_fx": _safe_ratio(scannet_fx, local_fx),
            "scannet_fy_over_local_fy": _safe_ratio(scannet_fy, local_fy),
            "scannet_fxy_over_local_fxy": _safe_ratio(scannet_fxy, local_fxy),
            "local_fx_over_scannet_fx": _safe_ratio(local_fx, scannet_fx),
            "local_fy_over_scannet_fy": _safe_ratio(local_fy, scannet_fy),
            "local_fxy_over_scannet_fxy": _safe_ratio(local_fxy, scannet_fxy),
            "scannet_fxy_over_ref_fxy": _safe_ratio(scannet_fxy, ref_fxy),
            "ref_fxy_over_scannet_fxy": _safe_ratio(ref_fxy, scannet_fxy),
            "local_fxy_over_ref_fxy": _safe_ratio(local_fxy, ref_fxy),
            "ref_fxy_over_local_fxy": _safe_ratio(ref_fxy, local_fxy),
            "local_scannet_reproj_error_px_median": local_scan_err["median"],
            "local_scannet_reproj_error_px_p90": local_scan_err["p90"],
            "local_pred_intrinsics_reproj_error_px_median": local_pred_err["median"],
            "local_pred_intrinsics_reproj_error_px_p90": local_pred_err["p90"],
            "ref_scannet_reproj_error_px_median": ref_scan_err["median"],
            "ref_scannet_reproj_error_px_p90": ref_scan_err["p90"],
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
        "scannet_fxy",
        "local_pred_fxy",
        "ref_pred_fxy",
        "local_scannet_reproj_error_px_median",
        "local_scannet_reproj_error_px_p90",
        "local_pred_intrinsics_reproj_error_px_median",
        "local_pred_intrinsics_reproj_error_px_p90",
        "ref_scannet_reproj_error_px_p90",
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
        "# v22.13 ref0 intrinsics proxy diagnostic",
        "",
        "Diagnostic-only: estimates OpenD4RT query-derived intrinsics from predicted xyz/uv and compares the resulting ratios with R23 eval-only ref0 depth scale. This does not define a method result.",
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
            "| scene | eval scale | S/local fxy | local/S fxy | local fxy | ScanNet fxy | local ScanNet reproj p90 | local pred-intr p90 | ref ScanNet reproj p90 | frames |",
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
                    fmt(row.get("scannet_fxy_over_local_fxy_median")),
                    fmt(row.get("local_fxy_over_scannet_fxy_median")),
                    fmt(row.get("local_pred_fxy_median")),
                    fmt(row.get("scannet_fxy_median")),
                    fmt(row.get("local_scannet_reproj_error_px_p90_median")),
                    fmt(row.get("local_pred_intrinsics_reproj_error_px_p90_median")),
                    fmt(row.get("ref_scannet_reproj_error_px_p90_median")),
                    fmt(row.get("frame_count")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v22 OpenD4RT intrinsics-from-query scale proxies.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v22_13_ref0_intrinsics_proxy_probe5")
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
    _write_csv(audit_root / "ref0_intrinsics_proxy_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "ref0_intrinsics_proxy_window_rows.csv", all_window_rows)
    _write_csv(audit_root / "ref0_intrinsics_proxy_scene_summary.csv", scene_summaries)
    _write_csv(audit_root / "ref0_intrinsics_proxy_candidate_errors.csv", candidate_rows)
    (audit_root / "ref0_intrinsics_proxy_frame_rows.json").write_text(json.dumps(_json_safe(all_frame_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_intrinsics_proxy_window_rows.json").write_text(json.dumps(_json_safe(all_window_rows), indent=2), encoding="utf-8")
    (audit_root / "ref0_intrinsics_proxy_scene_summary.json").write_text(json.dumps(_json_safe(scene_summaries), indent=2), encoding="utf-8")
    (audit_root / "ref0_intrinsics_proxy_candidate_errors.json").write_text(json.dumps(_json_safe(candidate_rows), indent=2), encoding="utf-8")
    _write_md(audit_root / "ref0_intrinsics_proxy.md", scene_summaries, candidate_rows)
    print(f"Wrote v22.13 ref0 intrinsics proxy diagnostic to {audit_root}")


if __name__ == "__main__":
    main()
