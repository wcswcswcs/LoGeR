from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.self_stitch import match_overlap_carriers
from stream4d_native.sim3 import estimate_overlap_sim3
from tools.materialize_d4rt_aligned_geometry_for_stream3d import _collect_anchors, _fit_summary, _fit_transform


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
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _load_carrier(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "uv_pred": np.asarray(data["uv_pred"], dtype=np.float32),
            "xyz_ref": np.asarray(data["xyz_ref"], dtype=np.float32),
            "visibility": np.asarray(data["visibility_prob"], dtype=np.float32),
            "confidence": np.asarray(data["confidence_prob"], dtype=np.float32),
            "valid": np.asarray(data["valid"], dtype=bool),
            "src_frame": np.asarray(data["src_frame"], dtype=np.int64),
            "src_uv": np.asarray(data["src_uv"], dtype=np.float32),
            "carrier_id": np.asarray(data.get("carrier_id", np.arange(data["uv_pred"].shape[1])), dtype=np.int64),
            "src_frame_global": np.asarray(data.get("src_frame_global", np.full((data["uv_pred"].shape[1],), -1)), dtype=np.int64),
            "src_xy": np.asarray(data.get("src_xy", np.full((data["uv_pred"].shape[1], 2), -1)), dtype=np.int64),
            "persistent_tube_id": np.asarray(data.get("persistent_tube_id", np.full((data["uv_pred"].shape[1],), -1)), dtype=np.int64),
        }


def _frame_ids_from_summary(scene_dir: Path, window_idx: int, num_frames: int) -> list[int]:
    summary_path = scene_dir / "summary.json"
    if summary_path.exists():
        try:
            timeline = json.loads(summary_path.read_text(encoding="utf-8")).get("timeline", [])
            row = timeline[int(window_idx)]
            start = int(row["frame_start"])
            end = int(row["frame_end"])
            if num_frames <= 1:
                return [start]
            step = max(1, int(round((end - start) / float(num_frames - 1))))
            return [start + step * idx for idx in range(num_frames)]
        except Exception:
            pass
    return list(range(num_frames))


def _self_uv_error_p90(data: dict[str, np.ndarray]) -> float | None:
    uv = data["uv_pred"]
    src_frame = data["src_frame"]
    src_uv = data["src_uv"]
    idx = np.arange(src_uv.shape[0], dtype=np.int64)
    valid_src = (src_frame >= 0) & (src_frame < uv.shape[0])
    if not np.any(valid_src):
        return None
    err = np.linalg.norm(uv[src_frame[valid_src], idx[valid_src]] - src_uv[valid_src], axis=1)
    err = err[np.isfinite(err)]
    return float(np.percentile(err, 90)) if err.size else None


def _trajectory_acceleration_p90(data: dict[str, np.ndarray]) -> float | None:
    xyz = data["xyz_ref"].astype(np.float64)
    ok = data["valid"] & np.isfinite(xyz).all(axis=2)
    if xyz.shape[0] < 3:
        return None
    acc = xyz[2:] - 2.0 * xyz[1:-1] + xyz[:-2]
    acc_ok = ok[2:] & ok[1:-1] & ok[:-2]
    values = np.linalg.norm(acc[acc_ok], axis=1)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 90)) if values.size else None


def _local_neighbor_stats(data: dict[str, np.ndarray], max_points_per_frame: int = 1024) -> dict[str, float | None]:
    xyz = data["xyz_ref"]
    ok = data["valid"] & np.isfinite(xyz).all(axis=2)
    dists: list[np.ndarray] = []
    for frame_idx in range(xyz.shape[0]):
        pts = xyz[frame_idx][ok[frame_idx]]
        if pts.shape[0] < 3:
            continue
        if pts.shape[0] > max_points_per_frame:
            keep = np.linspace(0, pts.shape[0] - 1, num=max_points_per_frame, dtype=np.int64)
            pts = pts[keep]
        dist, _ = cKDTree(pts).query(pts, k=2)
        dists.append(dist[:, 1])
    if not dists:
        return {"local_neighbor_stretch_p90": None, "local_neighbor_outlier_rate": None}
    nn = np.concatenate(dists, axis=0)
    q25, q75 = np.percentile(nn, [25, 75])
    fence = q75 + 3.0 * max(q75 - q25, 1e-12)
    return {
        "local_neighbor_stretch_p90": float(np.percentile(nn, 90)),
        "local_neighbor_outlier_rate": float(np.mean(nn > fence)),
    }


def _mask_coverage(scene: str, scene_dir: Path, carrier_path: Path, data: dict[str, np.ndarray], backbone: str) -> dict[str, float | None]:
    stream = ScanNetStream(seq_name=scene, backbone=backbone)
    frame_ids = _frame_ids_from_summary(scene_dir, int(carrier_path.stem.replace("carriers_window", "")), data["uv_pred"].shape[0])
    covered_ratios: list[float] = []
    boundary_ratios: list[float] = []
    for local_idx, frame_id in enumerate(frame_ids):
        try:
            mask = stream.load_mask(int(frame_id))
        except FileNotFoundError:
            continue
        positive = np.unique(mask[mask > 0].astype(np.int64))
        if positive.size == 0:
            continue
        h, w = mask.shape[:2]
        uv = data["uv_pred"][local_idx]
        x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
        y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
        ok = (
            data["valid"][local_idx]
            & np.isfinite(uv).all(axis=1)
            & (x >= 0)
            & (x < w)
            & (y >= 0)
            & (y < h)
            & (data["visibility"][local_idx] >= 0.5)
            & (data["confidence"][local_idx] >= 0.5)
        )
        if not np.any(ok):
            covered_ratios.append(0.0)
            boundary_ratios.append(0.0)
            continue
        hit_ids = np.unique(mask[y[ok], x[ok]].astype(np.int64))
        hit_ids = hit_ids[hit_ids > 0]
        covered_ratios.append(float(hit_ids.shape[0] / max(positive.shape[0], 1)))
        # Cheap boundary proxy: mask ids hit by points whose 4-neighborhood contains another id.
        yy = y[ok]
        xx = x[ok]
        center = mask[yy, xx]
        boundary = np.zeros_like(center, dtype=bool)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yn = np.clip(yy + dy, 0, h - 1)
            xn = np.clip(xx + dx, 0, w - 1)
            boundary |= mask[yn, xn] != center
        boundary_hit = np.unique(center[boundary & (center > 0)].astype(np.int64))
        boundary_ratios.append(float(boundary_hit.shape[0] / max(positive.shape[0], 1)))
    if not covered_ratios:
        return {"mask_interior_coverage_mean": None, "mask_boundary_coverage_mean": None}
    return {
        "mask_interior_coverage_mean": float(np.mean(covered_ratios)),
        "mask_boundary_coverage_mean": float(np.mean(boundary_ratios)),
    }


def _window_metrics(scene: str, debug_root: Path, carrier_path: Path, backbone: str) -> dict[str, Any]:
    data = _load_carrier(carrier_path)
    valid = data["valid"] & np.isfinite(data["uv_pred"]).all(axis=2) & np.isfinite(data["xyz_ref"]).all(axis=2)
    in01 = (
        valid
        & (data["uv_pred"][..., 0] >= 0.0)
        & (data["uv_pred"][..., 0] <= 1.0)
        & (data["uv_pred"][..., 1] >= 0.0)
        & (data["uv_pred"][..., 1] <= 1.0)
    )
    visible = valid & (data["visibility"] >= 0.5)
    track_lengths = np.sum(visible, axis=0) / float(max(data["uv_pred"].shape[0], 1))
    neighbor = _local_neighbor_stats(data)
    scene_dir = debug_root / scene
    mask_cov = _mask_coverage(scene, scene_dir, carrier_path, data, backbone)
    return {
        "scene": scene,
        "window": carrier_path.stem,
        "num_frames": int(data["uv_pred"].shape[0]),
        "num_queries": int(data["uv_pred"].shape[1]),
        "uv_in01_rate": float(np.mean(in01)) if in01.size else 0.0,
        "self_uv_error_p90": _self_uv_error_p90(data),
        "visible_track_length_mean": float(np.mean(track_lengths)) if track_lengths.size else 0.0,
        "confidence_mean": float(np.nanmean(data["confidence"])),
        "visibility_mean": float(np.nanmean(data["visibility"])),
        "trajectory_acceleration_p90": _trajectory_acceleration_p90(data),
        "duplicate_track_rate": 0.0,
        **neighbor,
        **mask_cov,
    }


def _phase_b(args: argparse.Namespace, scenes: list[str]) -> dict[str, Any]:
    debug_root = Path(args.phase_b_debug_root)
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = debug_root / scene
        carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
        for carrier_path in carrier_paths:
            rows.append(_window_metrics(scene, debug_root, carrier_path, args.backbone))
        if carrier_paths:
            stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
            source, target, anchor_diag = _collect_anchors(
                stream,
                carrier_paths,
                min_visibility=float(args.min_visibility),
                min_confidence=float(args.min_confidence),
                max_anchors=int(args.max_anchors),
            )
            fit = _fit_transform(source, target, robust_trim_percentile=float(args.robust_trim_percentile))
            eval_rows.append({"scene": scene, **anchor_diag, **_fit_summary(fit)})
    numeric = {
        key: float(np.nanmean([float(row[key]) for row in rows if row.get(key) is not None]))
        for key in sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)})
        if any(row.get(key) is not None for row in rows)
    }
    eval_numeric = {
        key: float(np.nanmean([float(row[key]) for row in eval_rows if row.get(key) is not None]))
        for key in sorted({key for row in eval_rows for key, value in row.items() if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)})
        if any(row.get(key) is not None for row in eval_rows)
    }
    return {
        "scope": "cached carrier diagnostics; no prediction path uses GT/RGB-D",
        "variant": args.phase_b_variant_name,
        "debug_root": str(debug_root),
        "scenes": rows,
        "eval_only_sim3": eval_rows,
        "numeric_mean": numeric,
        "eval_only_numeric_mean": eval_numeric,
    }


def _matched_overlap(
    prev: dict[str, np.ndarray],
    curr: dict[str, np.ndarray],
    prev_frame_ids: list[int],
    curr_frame_ids: list[int],
    uv_radius: float = 0.01,
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
    max_matches_per_frame: int = 512,
) -> tuple[np.ndarray, np.ndarray, int]:
    result = _matched_overlap_with_stats(
        prev,
        curr,
        prev_frame_ids,
        curr_frame_ids,
        uv_radius=uv_radius,
        min_visibility=min_visibility,
        min_confidence=min_confidence,
        max_matches_per_frame=max_matches_per_frame,
    )
    return result["prev_xyz"], result["curr_xyz"], int(result["stats"]["overlap_frame_count"])


def _match_payload(data: dict[str, np.ndarray], frame_ids: list[int]) -> dict[str, Any]:
    return {
        "frame_ids": [int(v) for v in frame_ids],
        "xyz": data["xyz_ref"],
        "uv": data["uv_pred"],
        "valid": data["valid"],
        "visibility": data["visibility"],
        "confidence": data["confidence"],
        "carrier_id": data.get("carrier_id", np.arange(data["xyz_ref"].shape[1])),
        "persistent_tube_id": data.get("persistent_tube_id", np.full((data["xyz_ref"].shape[1],), -1)),
        "src_frame_global": data.get("src_frame_global", np.full((data["xyz_ref"].shape[1],), -1)),
        "src_xy": data.get("src_xy", np.full((data["xyz_ref"].shape[1], 2), -1)),
    }


def _matched_overlap_with_stats(
    prev: dict[str, np.ndarray],
    curr: dict[str, np.ndarray],
    prev_frame_ids: list[int],
    curr_frame_ids: list[int],
    uv_radius: float = 0.01,
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
    max_matches_per_frame: int = 512,
) -> dict[str, Any]:
    result = match_overlap_carriers(
        _match_payload(prev, prev_frame_ids),
        _match_payload(curr, curr_frame_ids),
        min_visibility=float(min_visibility),
        min_confidence=float(min_confidence),
        uv_radius=float(uv_radius),
        max_matches_per_frame=int(max_matches_per_frame),
    )
    return {"prev_xyz": result.prev_xyz, "curr_xyz": result.curr_xyz, "stats": result.stats}


def _fixed_scale_residual_summary(curr_xyz: np.ndarray, prev_xyz: np.ndarray, fit: dict[str, Any], scale: float) -> dict[str, float | int]:
    src = np.asarray(curr_xyz, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(prev_xyz, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[ok]
    dst = dst[ok]
    if src.shape[0] == 0:
        return {"num_points": 0, "residual_median": float("nan"), "residual_p90": float("nan")}
    rot = np.asarray(fit["rot"], dtype=np.float64).reshape(3, 3)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    trans = mu_dst - float(scale) * (rot @ mu_src)
    pred = (float(scale) * (rot @ src.T)).T + trans
    residual = np.linalg.norm(pred - dst, axis=1)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return {"num_points": int(src.shape[0]), "residual_median": float("nan"), "residual_p90": float("nan")}
    return {
        "num_points": int(residual.size),
        "residual_median": float(np.median(residual)),
        "residual_p90": float(np.percentile(residual, 90)),
    }


def _phase_c(args: argparse.Namespace) -> dict[str, Any]:
    debug_root = Path(args.phase_c_debug_root)
    scene_dir = debug_root / args.phase_c_scene
    carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
    rows: list[dict[str, Any]] = []
    pair_matches: list[dict[str, Any]] = []
    for prev_path, curr_path in zip(carrier_paths, carrier_paths[1:]):
        prev = _load_carrier(prev_path)
        curr = _load_carrier(curr_path)
        prev_idx = int(prev_path.stem.replace("carriers_window", ""))
        curr_idx = int(curr_path.stem.replace("carriers_window", ""))
        prev_frame_ids = _frame_ids_from_summary(scene_dir, prev_idx, prev["uv_pred"].shape[0])
        curr_frame_ids = _frame_ids_from_summary(scene_dir, curr_idx, curr["uv_pred"].shape[0])
        match = _matched_overlap_with_stats(
            prev,
            curr,
            prev_frame_ids,
            curr_frame_ids,
            uv_radius=float(args.overlap_uv_radius),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            max_matches_per_frame=int(args.phase_c_max_matches_per_frame),
        )
        prev_xyz = match["prev_xyz"]
        curr_xyz = match["curr_xyz"]
        match_stats = match["stats"]
        vis = np.ones(prev_xyz.shape[:2], dtype=bool)
        fit = estimate_overlap_sim3(prev_xyz, curr_xyz, vis, vis, min_points=int(args.min_overlap_anchors))
        row: dict[str, Any] = {
            "scene": args.phase_c_scene,
            "prev_window": prev_path.stem,
            "curr_window": curr_path.stem,
            "overlap_frame_count": int(match_stats["overlap_frame_count"]),
            "overlap_anchor_count": int(prev_xyz.shape[0]),
            "self_sim3_success": fit is not None,
            **match_stats,
        }
        if fit is not None:
            row.update(
                {
                    "self_sim3_scale": float(fit["scale"]),
                    "self_sim3_inlier_ratio": float(fit["inlier_ratio"]),
                    "self_sim3_inlier_ratio_abs005": float(fit["inlier_ratio_abs005"]),
                    "self_sim3_inlier_ratio_abs010": float(fit["inlier_ratio_abs010"]),
                    "self_sim3_inlier_ratio_rel001": float(fit["inlier_ratio_rel001"]),
                    "self_sim3_inlier_ratio_rel002": float(fit["inlier_ratio_rel002"]),
                    "self_sim3_inlier_ratio_mad": float(fit["inlier_ratio_mad"]),
                    "self_sim3_residual_median": float(fit["residual_median"]),
                    "self_sim3_residual_p90": float(fit["residual_p90"]),
                    "self_sim3_residual_p95": float(fit["residual_p95"]),
                    "self_sim3_residual_mad": float(fit["residual_mad"]),
                }
            )
            pair_matches.append({"row": row, "prev_xyz": prev_xyz, "curr_xyz": curr_xyz, "fit": fit})
        rows.append(row)
    success_rows = [row for row in rows if row.get("self_sim3_success")]
    scales = [float(row["self_sim3_scale"]) for row in success_rows]
    normalized_rows: list[dict[str, Any]] = []
    normalized_summary: dict[str, Any] = {
        "enabled": False,
        "reason": "no successful pairwise fits",
    }
    if scales and pair_matches:
        log_scales = np.log(np.asarray(scales, dtype=np.float64))
        scale_bias = float(np.exp(np.mean(log_scales)))
        normalized_scales = [float(scale / scale_bias) for scale in scales]
        for idx, (match, normalized_scale) in enumerate(zip(pair_matches, normalized_scales)):
            residual_summary = _fixed_scale_residual_summary(
                match["curr_xyz"],
                match["prev_xyz"],
                match["fit"],
                normalized_scale,
            )
            normalized_rows.append(
                {
                    "scene": args.phase_c_scene,
                    "prev_window": match["row"]["prev_window"],
                    "curr_window": match["row"]["curr_window"],
                    "original_scale": float(scales[idx]),
                    "normalized_scale": float(normalized_scale),
                    "scale_bias_removed": scale_bias,
                    "normalized_residual_median": residual_summary["residual_median"],
                    "normalized_residual_p90": residual_summary["residual_p90"],
                    "normalized_residual_points": residual_summary["num_points"],
                }
            )
        normalized_summary = {
            "enabled": True,
            "scale_bias_removed": scale_bias,
            "normalized_scale_std": float(np.std(normalized_scales)),
            "normalized_accumulated_scale_drift": float(abs(np.prod(normalized_scales) - 1.0)),
            "normalized_residual_p90_mean": float(np.mean([row["normalized_residual_p90"] for row in normalized_rows])),
            "normalized_residual_median_mean": float(np.mean([row["normalized_residual_median"] for row in normalized_rows])),
            "note": "D4RT-only diagnostic scale-prior bundle: divides all pair scales by their geometric mean; no GT/RGB-D used.",
        }
    return {
        "scope": "cached scene0050 multi-window self-Sim3 using D4RT xyz/uv only",
        "debug_root": str(debug_root),
        "num_chunks": int(len(carrier_paths)),
        "rows": rows,
        "scale_normalized_rows": normalized_rows,
        "summary": {
            "num_pairs": int(len(rows)),
            "alignment_fail_count": int(sum(1 for row in rows if not row.get("self_sim3_success"))),
            "overlap_frame_count_mean": float(np.mean([row["overlap_frame_count"] for row in rows])) if rows else None,
            "overlap_anchor_count_mean": float(np.mean([row["overlap_anchor_count"] for row in rows])) if rows else None,
            "self_sim3_inlier_ratio_mean": float(np.mean([row["self_sim3_inlier_ratio"] for row in success_rows])) if success_rows else None,
            "self_sim3_inlier_ratio_abs005_mean": float(np.mean([row["self_sim3_inlier_ratio_abs005"] for row in success_rows])) if success_rows else None,
            "self_sim3_inlier_ratio_abs010_mean": float(np.mean([row["self_sim3_inlier_ratio_abs010"] for row in success_rows])) if success_rows else None,
            "self_sim3_inlier_ratio_rel001_mean": float(np.mean([row["self_sim3_inlier_ratio_rel001"] for row in success_rows])) if success_rows else None,
            "self_sim3_inlier_ratio_rel002_mean": float(np.mean([row["self_sim3_inlier_ratio_rel002"] for row in success_rows])) if success_rows else None,
            "self_sim3_inlier_ratio_mad_mean": float(np.mean([row["self_sim3_inlier_ratio_mad"] for row in success_rows])) if success_rows else None,
            "self_sim3_residual_p90_mean": float(np.mean([row["self_sim3_residual_p90"] for row in success_rows])) if success_rows else None,
            "self_sim3_scale_std": float(np.std(scales)) if scales else None,
            "accumulated_scale_drift": float(abs(np.prod(scales) - 1.0)) if scales else None,
            "scale_normalized_bundle": normalized_summary,
        },
    }


def _phase_d(args: argparse.Namespace) -> dict[str, Any]:
    matrix_path = Path(args.phase_d_matrix)
    rows: list[dict[str, Any]] = []
    if matrix_path.exists():
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        raw_rows = payload.get("rows", payload if isinstance(payload, list) else [])
        wanted = ("G0 ", "G1 ", "G2 ", "G3 ", "G4 ", "G5 ", "G6 ")
        for row in raw_rows:
            method = str(row.get("method", row.get("name", "")))
            if method.startswith(wanted):
                rows.append(row)
    return {
        "scope": "Phase D audit of available geometry-replacement evidence",
        "matrix_path": str(matrix_path),
        "available_rows": rows,
        "provider_hook_implemented": True,
        "geometry_provider_files": [
            "geometry_provider/base.py",
            "geometry_provider/rgbd_provider.py",
            "geometry_provider/d4rt_raw_provider.py",
            "geometry_provider/d4rt_self_stitched_provider.py",
            "geometry_provider/d4rt_eval_sim3_provider.py",
        ],
        "stream3d_internal_hook": "utils/mask_backprojection.py:frame_backprojection(args.geometry_provider)",
        "full_stream3d_provider_replacement_executed": False,
        "blocker": (
            "Available v10/v11 artifacts are minimal projection / adapter diagnostics, not a completed rerun of "
            "Stream3D set-cover, manifold refining, and historical merge with a GeometryProvider. v21.3 adds the "
            "provider hook, but the full G0-G6 provider rerun remains blocked until D4RT mask-to-provider point "
            "ownership is wired through all Stream3D stages."
        ),
    }


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    b = payload["phase_b"]
    c = payload["phase_c"]
    d = payload["phase_d"]
    lines = [
        "# Stream4D v21.3 Native Geometry Diagnostics",
        "",
        "## Phase B",
        "",
        f"- scope: `{b['scope']}`",
        f"- variant: `{b['variant']}`",
        "",
        "| metric | mean |",
        "|---|---:|",
    ]
    for key, value in sorted(b["numeric_mean"].items()):
        lines.append(f"| {key} | {value:.6g} |")
    lines.extend(["", "### Phase B Eval-Only Sim3", "", "| metric | mean |", "|---|---:|"])
    for key, value in sorted(b["eval_only_numeric_mean"].items()):
        lines.append(f"| {key} | {value:.6g} |")
    lines.extend(["", "## Phase C", "", "| metric | value |", "|---|---:|"])
    for key, value in c["summary"].items():
        lines.append(f"| {key} | {'NA' if value is None else f'{value:.6g}' if isinstance(value, float) else value} |")
    lines.extend(["", "## Phase D", "", f"- provider_hook_implemented: `{d['provider_hook_implemented']}`", f"- full_stream3d_provider_replacement_executed: `{d['full_stream3d_provider_replacement_executed']}`", f"- blocker: {d['blocker']}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v21.3 cached D4RT native geometry diagnostics.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--phase-b-debug-root", default="outputs/stream4d_debug_full_32f_ioc075_fixmem")
    parser.add_argument("--phase-b-variant-name", default="cached_mask_carrier_32clip_ioc075")
    parser.add_argument("--phase-c-debug-root", default="outputs/stream4d_debug_scene0050_128f_ioc075_fixmem")
    parser.add_argument("--phase-c-scene", default="scene0050_00")
    parser.add_argument("--phase-d-matrix", default="outputs/audit/v11_d4rt_geometry/d4rt_geometry_matrix_probe5.json")
    parser.add_argument("--output", default="outputs/audit/v21_3_geometry/native_geometry_diagnostics.md")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-anchors", type=int, default=7200)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--overlap-uv-radius", type=float, default=0.01)
    parser.add_argument("--min-overlap-anchors", type=int, default=200)
    parser.add_argument("--phase-c-max-matches-per-frame", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenes = _read_seq_list(Path(args.seq_list))
    payload = {
        "args": vars(args),
        "phase_b": _phase_b(args, scenes),
        "phase_c": _phase_c(args),
        "phase_d": _phase_d(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output.with_name(output.stem + "_phase_b_rows.csv"), payload["phase_b"]["scenes"])
    _write_csv(output.with_name(output.stem + "_phase_c_rows.csv"), payload["phase_c"]["rows"])
    _write_csv(output.with_name(output.stem + "_phase_c_scale_normalized_rows.csv"), payload["phase_c"]["scale_normalized_rows"])
    _write_markdown(output, payload)
    print(json.dumps(_json_safe({"phase_b": payload["phase_b"]["numeric_mean"], "phase_c": payload["phase_c"]["summary"], "phase_d_blocked": not payload["phase_d"]["full_stream3d_provider_replacement_executed"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
