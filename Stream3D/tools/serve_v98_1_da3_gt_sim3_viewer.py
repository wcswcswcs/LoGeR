#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd
import viser
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAM3D_ROOT))

from stream4d_native.sim3 import apply_sim3_to_xyz, fit_sim3_umeyama  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _read_points_csv(path: Path, scene_id: str, columns: tuple[str, str, str]) -> np.ndarray:
    df = pd.read_csv(path)
    if "scene_id" not in df.columns:
        raise ValueError(f"{path} has no scene_id column")
    df = df[df["scene_id"] == scene_id].copy()
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"{path} missing point columns: {missing}")
    if "surfel_valid" in df.columns:
        df = df[df["surfel_valid"].astype(str).str.lower().isin(["true", "1"])]
    if "stitch_valid" in df.columns:
        df = df[df["stitch_valid"].astype(str).str.lower().isin(["true", "1"])]
    pts = df.loc[:, list(columns)].to_numpy(dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def _read_gt_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"failed to read GT points from {path}")
    colors = np.asarray(pcd.colors)
    if colors.shape == points.shape and colors.size:
        colors_u8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        colors_u8 = np.full(points.shape, 185, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors_u8[finite]


def _sample_indices(count: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count <= 0 or count <= sample_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(count, size=sample_count, replace=False)
    idx.sort()
    return idx.astype(np.int64)


def _robust_radius(points: np.ndarray) -> float:
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return 1.0
    return max(float(np.percentile(dist, 90.0)), 1e-6)


def _pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = centered.T @ centered / max(int(points.shape[0]) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return axes


def _candidate_initial_rotations(source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    src_axes = _pca_axes(source)
    tgt_axes = _pca_axes(target)
    rotations: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for perm in permutations(range(3)):
        tgt_perm = tgt_axes[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            sign_mat = np.diag(np.asarray(signs, dtype=np.float64))
            rot = tgt_perm @ sign_mat @ src_axes.T
            if np.linalg.det(rot) > 0.0:
                rotations.append(rot.astype(np.float64))
    unique: list[np.ndarray] = []
    for rot in rotations:
        if not any(np.allclose(rot, existing, atol=1e-8) for existing in unique):
            unique.append(rot)
    return unique


def _residual_stats(dist: np.ndarray) -> dict[str, float]:
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return {key: float("nan") for key in ["mean", "p50", "p75", "p90", "p95", "max"]}
    return {
        "mean": float(np.mean(dist)),
        "p50": float(np.percentile(dist, 50.0)),
        "p75": float(np.percentile(dist, 75.0)),
        "p90": float(np.percentile(dist, 90.0)),
        "p95": float(np.percentile(dist, 95.0)),
        "max": float(np.max(dist)),
    }


def _fit_nn_diagnostic_sim3(
    source: np.ndarray,
    target_fit: np.ndarray,
    *,
    iterations: int,
    keep_ratio: float,
) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target_fit = np.asarray(target_fit, dtype=np.float64)
    if source.shape[0] < 4 or target_fit.shape[0] < 4:
        raise ValueError("at least four source and target points are required")

    target_tree = cKDTree(target_fit)
    source_radius = _robust_radius(source)
    target_radius = _robust_radius(target_fit)
    scale0 = float(target_radius / source_radius)
    src_center = np.median(source, axis=0)
    tgt_center = np.median(target_fit, axis=0)

    best: dict[str, Any] | None = None
    for start_id, rot0 in enumerate(_candidate_initial_rotations(source, target_fit)):
        scale = scale0
        rot = rot0
        trans = tgt_center - scale * (rot @ src_center)
        history: list[dict[str, Any]] = []
        failed = False
        for iteration in range(int(iterations)):
            aligned = apply_sim3_to_xyz(source, scale=scale, rot=rot, trans=trans).astype(np.float64)
            dist, nn_idx = target_tree.query(aligned, k=1)
            finite = np.isfinite(dist)
            if int(np.count_nonzero(finite)) < 4:
                failed = True
                break
            threshold = float(np.percentile(dist[finite], 100.0 * float(keep_ratio)))
            keep = finite & (dist <= threshold)
            if int(np.count_nonzero(keep)) < 4:
                keep = finite
            fit = fit_sim3_umeyama(source[keep], target_fit[nn_idx[keep]])
            scale = float(fit["scale"])
            rot = np.asarray(fit["rot"], dtype=np.float64)
            trans = np.asarray(fit["trans"], dtype=np.float64)
            history.append(
                {
                    "iteration": int(iteration),
                    "kept_pairs": int(np.count_nonzero(keep)),
                    "nn_distance_threshold": threshold,
                    "fit_anchor_residual_mean": float(np.mean(fit["residual"])),
                    "fit_anchor_residual_p90": float(np.percentile(fit["residual"], 90.0)),
                }
            )
        if failed:
            continue
        final_aligned = apply_sim3_to_xyz(source, scale=scale, rot=rot, trans=trans).astype(np.float64)
        final_dist, _ = target_tree.query(final_aligned, k=1)
        stats = _residual_stats(final_dist)
        candidate = {
            "start_id": int(start_id),
            "scale": float(scale),
            "rot": rot,
            "trans": trans,
            "rotation_det": float(np.linalg.det(rot)),
            "initial_scale": float(scale0),
            "history": history,
            "fit_sample_nn_residual": stats,
            "score": float(stats["p90"]),
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    if best is None:
        raise RuntimeError("diagnostic Sim3 nearest-neighbor fitting failed for all initial rotations")
    return best


def _prepare_payload(args: argparse.Namespace) -> dict[str, Any]:
    scene_id = args.scene_id
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    phase5_csv = Path(args.phase5_csv)
    phase3_csv = Path(args.phase3_csv)
    gt_ply = Path(args.gt_ply) if args.gt_ply else ROOT / "Stream3D" / "data" / "scannet" / "processed" / scene_id / f"{scene_id}_vh_clean_2.ply"
    phase5_points = _read_points_csv(phase5_csv, scene_id, ("xyz_x", "xyz_y", "xyz_z"))
    phase3_points = _read_points_csv(phase3_csv, scene_id, ("xyz_stitched_x", "xyz_stitched_y", "xyz_stitched_z"))
    gt_points, gt_colors = _read_gt_point_cloud(gt_ply)

    fit_idx = _sample_indices(gt_points.shape[0], int(args.fit_gt_sample_count), int(args.seed) + 17)
    viewer_idx = _sample_indices(gt_points.shape[0], int(args.viewer_gt_sample_count), int(args.seed) + 31)
    gt_fit = gt_points[fit_idx]
    gt_viewer = gt_points[viewer_idx]
    gt_viewer_colors = gt_colors[viewer_idx]

    sim3 = _fit_nn_diagnostic_sim3(
        phase5_points,
        gt_fit,
        iterations=int(args.fit_iterations),
        keep_ratio=float(args.fit_keep_ratio),
    )
    phase5_aligned = apply_sim3_to_xyz(
        phase5_points,
        scale=float(sim3["scale"]),
        rot=np.asarray(sim3["rot"], dtype=np.float64),
        trans=np.asarray(sim3["trans"], dtype=np.float64),
    ).astype(np.float32)
    phase3_aligned = apply_sim3_to_xyz(
        phase3_points,
        scale=float(sim3["scale"]),
        rot=np.asarray(sim3["rot"], dtype=np.float64),
        trans=np.asarray(sim3["trans"], dtype=np.float64),
    ).astype(np.float32)

    full_tree = cKDTree(gt_points)
    phase5_full_dist, _ = full_tree.query(phase5_aligned.astype(np.float64), k=1)
    phase3_full_dist, _ = full_tree.query(phase3_aligned.astype(np.float64), k=1)
    phase5_color = np.tile(np.asarray([[220, 45, 38]], dtype=np.uint8), (phase5_aligned.shape[0], 1))
    phase3_color = np.tile(np.asarray([[38, 107, 220]], dtype=np.uint8), (phase3_aligned.shape[0], 1))

    npz_path = output_root / f"{scene_id}_v98_da3_gt_sim3_viewer_points.npz"
    np.savez_compressed(
        npz_path,
        gt_points=gt_viewer.astype(np.float32),
        gt_colors=gt_viewer_colors.astype(np.uint8),
        v98_phase5_da3_surfels_sim3=phase5_aligned.astype(np.float32),
        v98_phase5_da3_surfels_colors=phase5_color,
        v98_phase3_da3_smoke_sim3=phase3_aligned.astype(np.float32),
        v98_phase3_da3_smoke_colors=phase3_color,
        v98_phase5_da3_surfels_raw=phase5_points.astype(np.float32),
        v98_phase3_da3_smoke_raw=phase3_points.astype(np.float32),
    )

    summary = {
        "viewer": "v98_1_da3_gt_sim3_viewer",
        "scene_id": scene_id,
        "diagnostic_only": True,
        "method_result_allowed": False,
        "sim3_alignment_kind": "diagnostic_nearest_neighbor_icp_umeyama_to_gt_mesh",
        "sim3_alignment_warning": (
            "No DA3-to-GT one-to-one anchors are available here. "
            "This Sim3 is fitted to nearest GT mesh points for visualization only; "
            "it is not method prediction evidence and must not be used as AP evidence."
        ),
        "source_files": {
            "gt_mesh": str(gt_ply),
            "v98_phase5_fused_surfel_csv": str(phase5_csv),
            "v98_phase3_stitched_da3_smoke_csv": str(phase3_csv),
        },
        "output_files": {
            "viewer_npz": str(npz_path),
            "summary_json": str(output_root / "summary.json"),
        },
        "counts": {
            "gt_mesh_point_count": int(gt_points.shape[0]),
            "gt_fit_sample_count": int(gt_fit.shape[0]),
            "gt_viewer_sample_count": int(gt_viewer.shape[0]),
            "v98_phase5_fused_surfel_count": int(phase5_points.shape[0]),
            "v98_phase3_stitched_da3_smoke_count": int(phase3_points.shape[0]),
        },
        "diagnostic_sim3": {
            "scale": float(sim3["scale"]),
            "rotation_det": float(sim3["rotation_det"]),
            "translation": np.asarray(sim3["trans"], dtype=np.float64),
            "rotation": np.asarray(sim3["rot"], dtype=np.float64),
            "initial_scale": float(sim3["initial_scale"]),
            "selected_start_id": int(sim3["start_id"]),
            "fit_iterations": int(args.fit_iterations),
            "fit_keep_ratio": float(args.fit_keep_ratio),
            "fit_history": sim3["history"],
            "fit_sample_phase5_to_gt_nn_residual": sim3["fit_sample_nn_residual"],
            "full_gt_phase5_to_gt_nn_residual": _residual_stats(phase5_full_dist),
            "full_gt_phase3_smoke_to_gt_nn_residual": _residual_stats(phase3_full_dist),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "viewer_npz": npz_path,
        "gt_points": gt_viewer.astype(np.float32),
        "gt_colors": gt_viewer_colors.astype(np.uint8),
        "phase5_points": phase5_aligned.astype(np.float32),
        "phase5_colors": phase5_color,
        "phase3_points": phase3_aligned.astype(np.float32),
        "phase3_colors": phase3_color,
    }


def serve(args: argparse.Namespace) -> dict[str, Any]:
    payload = _prepare_payload(args)
    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=float(args.grid_width),
        height=float(args.grid_width),
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    gt_handle = server.scene.add_point_cloud(
        "/GT ScanNet mesh sample",
        points=payload["gt_points"],
        colors=payload["gt_colors"],
        point_size=float(args.gt_point_size),
        point_shape="circle",
        visible=True,
    )
    phase5_handle = server.scene.add_point_cloud(
        "/v98 DA3 Phase5 fused surfels Sim3-aligned",
        points=payload["phase5_points"],
        colors=payload["phase5_colors"],
        point_size=float(args.phase5_point_size),
        point_shape="circle",
        visible=True,
    )
    phase3_handle = server.scene.add_point_cloud(
        "/v98 DA3 Phase3 stitch smoke Sim3-aligned",
        points=payload["phase3_points"],
        colors=payload["phase3_colors"],
        point_size=float(args.phase3_point_size),
        point_shape="circle",
        visible=False,
    )

    gt_toggle = server.gui.add_checkbox("GT ScanNet mesh sample", True)
    phase5_toggle = server.gui.add_checkbox("v98 DA3 Phase5 surfels Sim3", True)
    phase3_toggle = server.gui.add_checkbox("v98 DA3 Phase3 smoke Sim3", False)

    @gt_toggle.on_update
    def _(_: Any) -> None:
        gt_handle.visible = bool(gt_toggle.value)

    @phase5_toggle.on_update
    def _(_: Any) -> None:
        phase5_handle.visible = bool(phase5_toggle.value)

    @phase3_toggle.on_update
    def _(_: Any) -> None:
        phase3_handle.visible = bool(phase3_toggle.value)

    status = {
        "viewer": "v98_1_da3_gt_sim3_viewer",
        "pid": int(os.getpid()),
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{args.port}",
        "scene_id": args.scene_id,
        "viewer_npz": str(payload["viewer_npz"]),
        "summary_json": str(payload["summary_path"]),
        "diagnostic_only": True,
        "layers": [
            "GT ScanNet mesh sample",
            "v98 DA3 Phase5 fused surfels Sim3-aligned",
            "v98 DA3 Phase3 stitch smoke Sim3-aligned",
        ],
        "counts": payload["summary"]["counts"],
        "diagnostic_sim3": {
            "scale": payload["summary"]["diagnostic_sim3"]["scale"],
            "rotation_det": payload["summary"]["diagnostic_sim3"]["rotation_det"],
            "full_gt_phase5_to_gt_nn_residual": payload["summary"]["diagnostic_sim3"]["full_gt_phase5_to_gt_nn_residual"],
            "full_gt_phase3_smoke_to_gt_nn_residual": payload["summary"]["diagnostic_sim3"]["full_gt_phase3_smoke_to_gt_nn_residual"],
        },
    }
    if args.status_json:
        status_path = Path(args.status_json)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True, default=_json_default), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if float(args.smoke_seconds) > 0:
        deadline = time.time() + float(args.smoke_seconds)
        while time.time() < deadline:
            time.sleep(0.25)
        server.stop()
        return status

    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Viser GT/v98-DA3 geometry viewer with diagnostic Sim3 alignment.")
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--phase5-csv", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase5_fused_surfel" / "fused_surfel_rows.csv"))
    parser.add_argument("--phase3-csv", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase3_da3_stitch" / "stitched_da3_point_rows.csv"))
    parser.add_argument("--gt-ply", default="")
    parser.add_argument("--output-root", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_1_da3_gt_sim3_viewer_scene0050"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--seed", type=int, default=9801050)
    parser.add_argument("--fit-gt-sample-count", type=int, default=60000)
    parser.add_argument("--viewer-gt-sample-count", type=int, default=100000)
    parser.add_argument("--fit-iterations", type=int, default=8)
    parser.add_argument("--fit-keep-ratio", type=float, default=0.75)
    parser.add_argument("--gt-point-size", type=float, default=0.012)
    parser.add_argument("--phase5-point-size", type=float, default=0.026)
    parser.add_argument("--phase3-point-size", type=float, default=0.055)
    parser.add_argument("--grid-width", type=float, default=8.0)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
