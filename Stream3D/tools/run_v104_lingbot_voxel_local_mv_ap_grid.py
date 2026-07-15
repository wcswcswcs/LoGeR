#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v103_phase6_mask_clustering_local_object_birth as base  # noqa: E402


PHASE_ID = "v104_lingbot_map_only_phase9_voxel_local_mv_ap_grid"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v104_lingbot_map_only_phase9_voxel_local_mv_ap_grid"
DEFAULT_PHASE5_ROOT = STREAM3D_ROOT / "outputs/audit/v104_lingbot_map_only_phase8_voxel_affinity_features"


def _threshold_tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p").rstrip("0").rstrip("p")


def _variants(thresholds: list[float], topks: list[int], min_frames: int) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = [
        {
            "variant_id": "L0_singleton_all_supported_min1_diagnostic",
            "clusterer": "singleton_masks_diagnostic",
            "pair_affinity_mode": "static_feature_cosine",
            "threshold": 1.01,
            "topk_per_mask": 0,
            "min_object_frames": 1,
            "use_cannot_link": True,
            "node_policy": "all_supported",
            "emit_policy": "all_supported",
        }
    ]
    policies = [
        ("obj", "object_like", "default", "all_node_same_frame", "member_broad_risk"),
        ("allpref", "all_supported", "prefer_object_like", "specific_non_broad_same_frame", "selected_broad_risk"),
        ("nonbroad", "supported_non_broad", "non_broad_only_skip", "specific_non_broad_same_frame", "selected_broad_risk"),
    ]
    for threshold in thresholds:
        for topk in topks:
            for tag, node_policy, emit_policy, cannot_policy, score_policy in policies:
                variants.append(
                    {
                        "variant_id": f"L1_{tag}_static_tau{_threshold_tag(threshold)}_top{int(topk)}_min{int(min_frames)}",
                        "clusterer": "constrained_union_find",
                        "pair_affinity_mode": "static_feature_cosine",
                        "threshold": float(threshold),
                        "topk_per_mask": int(topk),
                        "min_object_frames": int(min_frames),
                        "use_cannot_link": True,
                        "cannot_link_policy": cannot_policy,
                        "node_policy": node_policy,
                        "emit_policy": emit_policy,
                        "score_policy": score_policy,
                    }
                )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v104 LingBot voxel-calibrated local MV_AP grid via v103 Phase6 evaluator.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(base.DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(base.DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--baseline-rows", default=str(base.DEFAULT_BASELINE_ROWS))
    parser.add_argument("--thresholds", default="0.02,0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--topks", default="4,8,16")
    parser.add_argument("--min-object-frames", type=int, default=2)
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    args = parser.parse_args()

    thresholds = [float(v.strip()) for v in str(args.thresholds).split(",") if v.strip()]
    topks = [int(v.strip()) for v in str(args.topks).split(",") if v.strip()]
    base.PHASE_ID = PHASE_ID
    base.VARIANTS = _variants(thresholds, topks, int(args.min_object_frames))
    sys.argv = [
        str(Path(__file__)),
        "--output-root",
        str(args.output_root),
        "--phase5-root",
        str(args.phase5_root),
        "--scene0011-phase2-root",
        str(args.scene0011_phase2_root),
        "--scene0050-phase2-root",
        str(args.scene0050_phase2_root),
        "--baseline-rows",
        str(args.baseline_rows),
        "--min-pred-pixels",
        str(args.min_pred_pixels),
        "--min-gt-pixels",
        str(args.min_gt_pixels),
        "--pair-batch-size",
        str(args.pair_batch_size),
        "--cupy-device-id",
        str(args.cupy_device_id),
    ]
    if args.disable_cupy_iou:
        sys.argv.append("--disable-cupy-iou")
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
