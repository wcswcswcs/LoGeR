#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from build_v98_1_da3_variant_geometry_quality import (  # noqa: E402
    _chamfer_metrics,
    _evaluate_variant,
    _filter_gt_to_input_visible,
    _metric_row,
    _parse_da3_log,
    _percentiles,
    _write_csv,
    _write_json,
)
from serve_v98_1_da3_gt_dense_rgb_sim3_viewer import (  # noqa: E402
    _json_default,
    _load_da3_manifest,
    _read_gt_point_cloud,
    _sample_indices,
)


DEFAULT_PHASE1 = ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase1_provider_contract"


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scene_root = ROOT / "Stream3D" / "data" / "scannet" / "processed" / args.scene_id
    gt_ply = Path(args.gt_ply) if args.gt_ply else scene_root / f"{args.scene_id}_vh_clean_2.ply"
    gt_points_full, gt_colors_full = _read_gt_point_cloud(gt_ply)
    manifest = _load_da3_manifest(Path(args.da3_manifest))
    gt_visibility_filter: dict[str, Any] = {"mode": "full_gt_no_input_visibility_filter"}
    if args.gt_filter == "input_visible":
        visible_mask, gt_visibility_filter = _filter_gt_to_input_visible(
            gt_points=gt_points_full,
            manifest=manifest,
            scene_root=scene_root,
            depth_scale=float(args.scannet_depth_scale),
            depth_abs_tolerance=float(args.gt_visible_depth_abs_tolerance),
            depth_rel_tolerance=float(args.gt_visible_depth_rel_tolerance),
            min_observations=int(args.gt_visible_min_observations),
            batch_size=int(args.gt_visible_batch_size),
        )
        if int(np.count_nonzero(visible_mask)) < 4:
            raise RuntimeError("input-visible GT filter left fewer than 4 GT points")
        gt_points = gt_points_full[visible_mask]
        gt_colors = gt_colors_full[visible_mask]
    else:
        gt_points = gt_points_full
        gt_colors = gt_colors_full

    gt_tree = cKDTree(gt_points)
    gt_viewer_idx = _sample_indices(gt_points.shape[0], int(args.viewer_gt_sample_count), int(args.seed) + 31)
    scannet_pose_root = scene_root / "pose"
    variant_key = str(args.variant_key)
    variant = {
        "display_name": str(args.display_name),
        "model": str(args.model_name),
        "repo_id": str(args.repo_id),
        "root": Path(args.da3_root),
        "log": Path(args.da3_log),
    }
    row = _evaluate_variant(
        variant_key=variant_key,
        variant=variant,
        manifest=manifest,
        gt_points=gt_points,
        gt_tree=gt_tree,
        scannet_pose_root=scannet_pose_root,
        args=args,
    )
    payload = row.pop("viewer_payload")
    npz_payload: dict[str, Any] = {
        "gt_points": gt_points[gt_viewer_idx].astype(np.float32),
        "gt_colors": gt_colors[gt_viewer_idx].astype(np.uint8),
        f"{variant_key}_pose_points": payload["pose_points"],
        f"{variant_key}_surface_points": payload["surface_points"],
        f"{variant_key}_colors": payload["colors"],
    }
    csv_rows = [
        _metric_row(variant_key, variant, "pose_orientation_sim3", row["pose_orientation_sim3"]["geometry_metrics"]),
        _metric_row(variant_key, variant, "surface_refined_sim3", row["surface_refined_sim3"]["geometry_metrics"]),
    ]
    npz_path = output_root / f"{args.scene_id}_da3_single_geometry_viewer_points.npz"
    csv_path = output_root / "geometry_quality_metrics.csv"
    summary_path = output_root / "geometry_quality_summary.json"
    np.savez_compressed(npz_path, **npz_payload)
    _write_csv(csv_path, csv_rows)
    summary = {
        "scene_id": args.scene_id,
        "diagnostic_only": True,
        "method_result_allowed": False,
        "metric_note": (
            "Single-scene DA3 diagnostic base for Viser. surface_refined_sim3 uses GT mesh nearest-neighbor refinement "
            "for geometry visualization/diagnostic comparison, not method prediction evidence."
        ),
        "input": {
            "manifest": str(args.da3_manifest),
            "manifest_frame_count": int(manifest.shape[0]),
            "frame_id_min": int(manifest["frame_id"].min()),
            "frame_id_max": int(manifest["frame_id"].max()),
            "stride_frame_id": int(manifest["frame_id"].iloc[1] - manifest["frame_id"].iloc[0]) if manifest.shape[0] > 1 else None,
            "da3_root": str(args.da3_root),
            "da3_log": str(args.da3_log),
        },
        "gt": {
            "gt_ply": str(gt_ply),
            "gt_filter": str(args.gt_filter),
            "full_gt_point_count": int(gt_points_full.shape[0]),
            "eval_gt_point_count": int(gt_points.shape[0]),
            "viewer_gt_point_count": int(gt_viewer_idx.shape[0]),
            "full_gt_z_stats_m": _percentiles(gt_points_full[:, 2]),
            "eval_gt_z_stats_m": _percentiles(gt_points[:, 2]),
            "visibility_filter": gt_visibility_filter,
        },
        "outputs": {
            "summary_json": str(summary_path),
            "metrics_csv": str(csv_path),
            "viewer_npz": str(npz_path),
        },
        "variants": [row],
        "csv_rows": csv_rows,
        "da3_run_log": _parse_da3_log(Path(args.da3_log)),
    }
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {"summary_json": str(summary_path), "metrics_csv": str(csv_path), "viewer_npz": str(npz_path)},
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a generic one-DA3-root geometry base summary for v98.1 Viser.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--da3-root", required=True)
    parser.add_argument("--da3-manifest", required=True)
    parser.add_argument("--da3-log", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant-key", default="da3_streaming_full")
    parser.add_argument("--display-name", default="DA3-Streaming full")
    parser.add_argument("--model-name", default="DA3-SMALL")
    parser.add_argument("--repo-id", default="depth-anything/DA3-SMALL")
    parser.add_argument("--gt-ply", default="")
    parser.add_argument("--gt-filter", choices=["full", "input_visible"], default="input_visible")
    parser.add_argument("--scannet-depth-scale", type=float, default=1000.0)
    parser.add_argument("--gt-visible-depth-abs-tolerance", type=float, default=0.08)
    parser.add_argument("--gt-visible-depth-rel-tolerance", type=float, default=0.03)
    parser.add_argument("--gt-visible-min-observations", type=int, default=1)
    parser.add_argument("--gt-visible-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=9801098)
    parser.add_argument("--da3-dense-step", type=int, default=8)
    parser.add_argument("--da3-conf-min", type=float, default=0.0)
    parser.add_argument("--surface-fit-sample-count", type=int, default=60000)
    parser.add_argument("--surface-fit-iterations", type=int, default=8)
    parser.add_argument("--surface-fit-keep-ratio", type=float, default=0.90)
    parser.add_argument("--viewer-gt-sample-count", type=int, default=180000)
    parser.add_argument("--viewer-da3-sample-count", type=int, default=120000)
    parser.add_argument("--fscore-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    build(parser.parse_args())


if __name__ == "__main__":
    main()
