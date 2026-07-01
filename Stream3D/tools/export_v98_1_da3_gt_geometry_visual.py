#!/usr/bin/env python3
"""Export a one-scene v98.1 DA3-vs-GT geometry visualization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def _read_xyz_rows(path: Path, scene_id: str, keys: tuple[str, str, str]) -> np.ndarray:
    pts: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scene_id") != scene_id:
                continue
            xyz = [_as_float(row, key) for key in keys]
            if all(np.isfinite(xyz)):
                pts.append(xyz)
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _sample(points: np.ndarray, colors: np.ndarray | None, count: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if points.shape[0] <= count:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(points.shape[0], size=count, replace=False))
    sampled_colors = colors[idx] if colors is not None else None
    return points[idx], sampled_colors


def _normalize(points: np.ndarray, x_offset: float) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    centered = points - np.median(points, axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(centered, axis=1), 95)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    out = centered / scale
    out[:, 0] += x_offset
    return out


def _bbox(points: np.ndarray) -> dict[str, Any]:
    if points.size == 0:
        return {"count": 0}
    return {
        "count": int(points.shape[0]),
        "min": points.min(axis=0).astype(float).tolist(),
        "max": points.max(axis=0).astype(float).tolist(),
        "median": np.median(points, axis=0).astype(float).tolist(),
    }


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if colors.dtype != np.uint8:
        colors = np.asarray(np.clip(colors, 0, 255), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(
                f"{float(xyz[0]):.7f} {float(xyz[1]):.7f} {float(xyz[2]):.7f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def _rgb_strings(colors: np.ndarray, alpha: float) -> list[str]:
    return [f"rgba({int(r)},{int(g)},{int(b)},{alpha})" for r, g, b in colors]


def _trace(points: np.ndarray, name: str, color: str | list[str], size: float, opacity: float) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0] if points.size else [],
        y=points[:, 1] if points.size else [],
        z=points[:, 2] if points.size else [],
        mode="markers",
        marker={"size": size, "color": color, "opacity": opacity},
        name=name,
    )


def _scatter2d(ax: Any, points: np.ndarray, dims: tuple[int, int], color: str, label: str, size: float, alpha: float) -> None:
    if points.size == 0:
        return
    ax.scatter(points[:, dims[0]], points[:, dims[1]], s=size, c=color, alpha=alpha, label=label, linewidths=0)


def _write_quicklook(
    path: Path,
    scene_id: str,
    gt_sample: np.ndarray,
    surfels: np.ndarray,
    smoke: np.ndarray,
    gt_norm: np.ndarray,
    surfel_norm: np.ndarray,
    smoke_norm: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    panels = [
        (axes[0, 0], gt_sample, surfels, smoke, (0, 1), "raw XY"),
        (axes[0, 1], gt_sample, surfels, smoke, (0, 2), "raw XZ"),
        (axes[1, 0], gt_norm, surfel_norm, smoke_norm, (0, 1), "normalized side-by-side XY"),
        (axes[1, 1], gt_norm, surfel_norm, smoke_norm, (0, 2), "normalized side-by-side XZ"),
    ]
    for ax, gt, sf, sm, dims, title in panels:
        _scatter2d(ax, gt, dims, "#8f8f8f", "GT mesh sample", 0.5, 0.22)
        _scatter2d(ax, sf, dims, "#e34833", "v98 Phase5 fused DA3 surfels", 4.0, 0.9)
        _scatter2d(ax, sm, dims, "#2383ff", "Phase3 DA3 smoke", 14.0, 0.95)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.35)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False)
    fig.suptitle(
        f"{scene_id}: v98 DA3 geometry vs GT geometry\n"
        "Raw plots show different coordinate frames; normalized plots are shape-only diagnostics.",
        y=0.94,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def export(scene_id: str, output_root: Path, gt_sample_count: int, seed: int) -> dict[str, Any]:
    phase3_path = ROOT / "Stream3D/outputs/audit/v98_phase3_da3_stitch/stitched_da3_point_rows.csv"
    phase5_path = ROOT / "Stream3D/outputs/audit/v98_phase5_fused_surfel/fused_surfel_rows.csv"
    mesh_path = ROOT / "Stream3D/data/scannet/processed" / scene_id / f"{scene_id}_vh_clean_2.ply"
    if not phase3_path.is_file():
        raise FileNotFoundError(phase3_path)
    if not phase5_path.is_file():
        raise FileNotFoundError(phase5_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    smoke = _read_xyz_rows(phase3_path, scene_id, ("xyz_stitched_x", "xyz_stitched_y", "xyz_stitched_z"))
    surfels = _read_xyz_rows(phase5_path, scene_id, ("xyz_x", "xyz_y", "xyz_z"))
    cloud = o3d.io.read_point_cloud(str(mesh_path))
    gt_points = np.asarray(cloud.points, dtype=np.float32)
    gt_colors = np.asarray(cloud.colors, dtype=np.float32)
    if gt_colors.shape != gt_points.shape:
        gt_rgb = np.full(gt_points.shape, 180, dtype=np.uint8)
    else:
        gt_rgb = np.asarray(np.clip(gt_colors, 0, 1) * 255, dtype=np.uint8)
    gt_sample, gt_sample_rgb = _sample(gt_points, gt_rgb, gt_sample_count, seed)
    assert gt_sample_rgb is not None

    output_root.mkdir(parents=True, exist_ok=True)
    gt_ply = output_root / f"{scene_id}_gt_mesh_points_sampled_{gt_sample.shape[0]}.ply"
    surfel_ply = output_root / f"{scene_id}_v98_phase5_fused_surfel_da3_local.ply"
    smoke_ply = output_root / f"{scene_id}_v98_phase3_stitched_da3_smoke_local.ply"
    _write_ply(gt_ply, gt_sample, gt_sample_rgb)
    _write_ply(surfel_ply, surfels, np.tile(np.array([[235, 70, 50]], dtype=np.uint8), (surfels.shape[0], 1)))
    _write_ply(smoke_ply, smoke, np.tile(np.array([[40, 130, 255]], dtype=np.uint8), (smoke.shape[0], 1)))

    gt_norm = _normalize(gt_sample, -2.2)
    surfel_norm = _normalize(surfels, 0.0)
    smoke_norm = _normalize(smoke, 2.2)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Raw coordinates: GT ScanNet world vs v98 DA3 local",
            "Shape-only normalized side-by-side (not metric alignment)",
        ),
    )
    fig.add_trace(_trace(gt_sample, "GT mesh sample (ScanNet world)", _rgb_strings(gt_sample_rgb, 0.35), 1.4, 0.35), row=1, col=1)
    fig.add_trace(_trace(surfels, "v98 Phase5 fused DA3 surfels (local)", "rgb(235,70,50)", 3.0, 0.92), row=1, col=1)
    fig.add_trace(_trace(smoke, "v98 Phase3 DA3 stitch smoke (local)", "rgb(40,130,255)", 5.0, 0.95), row=1, col=1)
    fig.add_trace(_trace(gt_norm, "GT normalized", "rgba(150,150,150,0.45)", 1.5, 0.45), row=1, col=2)
    fig.add_trace(_trace(surfel_norm, "Phase5 surfels normalized", "rgb(235,70,50)", 3.0, 0.92), row=1, col=2)
    fig.add_trace(_trace(smoke_norm, "Phase3 smoke normalized", "rgb(40,130,255)", 5.0, 0.95), row=1, col=2)
    fig.update_layout(
        title=(
            f"{scene_id}: v98 DA3 geometry vs GT geometry<br>"
            "Red = v98 Phase5 fused DA3 surfels used by method, Blue = Phase3 DA3 stitch smoke, Gray/RGB = GT mesh sample"
        ),
        height=780,
        width=1500,
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
    )
    fig.update_scenes(aspectmode="data")
    html_path = output_root / f"{scene_id}_v98_da3_vs_gt_geometry.html"
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    quicklook_path = output_root / f"{scene_id}_v98_da3_vs_gt_geometry_quicklook.png"
    _write_quicklook(quicklook_path, scene_id, gt_sample, surfels, smoke, gt_norm, surfel_norm, smoke_norm)

    summary = {
        "scene_id": scene_id,
        "html": html_path.relative_to(ROOT).as_posix(),
        "quicklook_png": quicklook_path.relative_to(ROOT).as_posix(),
        "gt_mesh_path": mesh_path.relative_to(ROOT).as_posix(),
        "gt_sample_ply": gt_ply.relative_to(ROOT).as_posix(),
        "phase5_fused_surfel_ply": surfel_ply.relative_to(ROOT).as_posix(),
        "phase3_stitched_smoke_ply": smoke_ply.relative_to(ROOT).as_posix(),
        "phase5_fused_surfel_source": phase5_path.relative_to(ROOT).as_posix(),
        "phase3_stitched_smoke_source": phase3_path.relative_to(ROOT).as_posix(),
        "gt_sample_seed": seed,
        "coordinate_frame_note": (
            "GT points are ScanNet world/mesh coordinates. v98 DA3 Phase5 and Phase3 points are DA3 local/stitch coordinates. "
            "The raw plot intentionally does not claim metric alignment; the normalized subplot is shape-only diagnostic."
        ),
        "counts": {
            "gt_mesh_point_count": int(gt_points.shape[0]),
            "gt_sample_point_count": int(gt_sample.shape[0]),
            "v98_phase5_fused_surfel_count": int(surfels.shape[0]),
            "v98_phase3_stitched_smoke_count": int(smoke.shape[0]),
        },
        "bounds": {
            "gt_sample": _bbox(gt_sample),
            "v98_phase5_fused_surfel_da3_local": _bbox(surfels),
            "v98_phase3_stitched_da3_smoke_local": _bbox(smoke),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument(
        "--output-root",
        default="Stream3D/outputs/audit/v98_1_da3_gt_geometry_visual_scene0050",
    )
    parser.add_argument("--gt-sample-count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=9801050)
    args = parser.parse_args()
    summary = export(args.scene_id, ROOT / args.output_root, args.gt_sample_count, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
