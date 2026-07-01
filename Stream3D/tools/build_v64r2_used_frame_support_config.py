from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.v64r2_visible_support_utils import (
    frame_ids_from_debug_root,
    json_safe,
    read_seq_list,
    scene_points_from_stream,
    visible_support_point_ids,
)


def _copy_predictions(root: Path, input_config: str, output_config: str, scenes: list[str]) -> None:
    src = root / "data" / "prediction" / f"{input_config}_class_agnostic"
    dst = root / "data" / "prediction" / f"{output_config}_class_agnostic"
    dst.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        src_file = src / f"{scene}.npz"
        if not src_file.exists():
            raise FileNotFoundError(f"Missing prediction file: {src_file}")
        shutil.copy2(src_file, dst / src_file.name)


def _prediction_stats(root: Path, config: str, scene: str, support_ids: np.ndarray) -> dict[str, Any]:
    pred = np.load(root / "data" / "prediction" / f"{config}_class_agnostic" / f"{scene}.npz")
    masks = np.asarray(pred["pred_masks"], dtype=bool)
    union = np.flatnonzero(masks.any(axis=1)).astype(np.int64) if masks.shape[1] else np.empty((0,), dtype=np.int64)
    support_set = set(int(v) for v in support_ids.tolist())
    union_set = set(int(v) for v in union.tolist())
    return {
        "pred_instance_count": int(masks.shape[1]),
        "prediction_union_count": int(union.shape[0]),
        "support_count": int(support_ids.shape[0]),
        "prediction_union_ratio": float(union.shape[0] / max(masks.shape[0], 1)),
        "support_ratio": float(support_ids.shape[0] / max(masks.shape[0], 1)),
        "prediction_union_inside_support_count": int(len(union_set.intersection(support_set))),
        "prediction_union_inside_support_ratio": float(len(union_set.intersection(support_set)) / max(len(union_set), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a v64r2 AP config whose pre_points are used-frame depth/pose visible support.")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--audit-root", default="outputs/audit/v64r2_used_frame_support")
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--mask-positive-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = Path(".").resolve()
    scenes = read_seq_list(Path(args.seq_list))
    _copy_predictions(root, args.input_config, args.output_config, scenes)

    tmp_dir = root / "data" / "TMP" / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        stream = ScanNetStream(seq_name=scene, backbone="Cropformer")
        scene_points = scene_points_from_stream(stream)
        frame_ids = frame_ids_from_debug_root(args.debug_root, scene)
        support_ids, support_diag = visible_support_point_ids(
            stream,
            scene_points,
            frame_ids,
            pixel_stride=int(args.pixel_stride),
            nn_radius=float(args.nn_radius),
            mask_positive_only=bool(args.mask_positive_only),
        )
        np.save(tmp_dir / f"{scene}_pre_points.npy", support_ids.astype(np.int64))
        row = {
            "scene": scene,
            "input_config": args.input_config,
            "output_config": args.output_config,
            "debug_root": args.debug_root,
            "frame_ids": frame_ids,
            **support_diag,
            **_prediction_stats(root, args.output_config, scene, support_ids),
        }
        rows.append(row)

    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_config": args.input_config,
        "output_config": args.output_config,
        "debug_root": args.debug_root,
        "seq_list": args.seq_list,
        "pre_points_policy": "used_frame_depth_pose_visible_mask_support",
        "pixel_stride": int(args.pixel_stride),
        "nn_radius": float(args.nn_radius),
        "mask_positive_only": bool(args.mask_positive_only),
        "rows": rows,
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)})
            if any(row.get(key) is not None for row in rows)
        },
    }
    (audit_root / f"{args.output_config}_used_frame_support_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config, args.debug_root],
        pre_points_policy="used_frame_depth_pose_visible_mask_support",
        support_policy="copied_predictions_fixed_to_used_frame_depth_pose_support",
        notes=(
            "Diagnostic AP config copied from an existing prediction config. "
            "Only data/TMP pre_points are replaced by mesh vertices visible from the actual used frames via ScanNet depth/pose/mask."
        ),
        extra={
            "algorithm": "build_v64r2_used_frame_support_config",
            "input_config": args.input_config,
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "debug_root": args.debug_root,
            "support_pixel_stride": int(args.pixel_stride),
            "support_nn_radius": float(args.nn_radius),
            "support_mask_positive_only": bool(args.mask_positive_only),
            "uses_rgbd_for_evaluation_support": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "summary_path": str(audit_root / f"{args.output_config}_used_frame_support_summary.json"),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root, pred_suffix="class_agnostic")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
