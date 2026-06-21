from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_native_typed_assembly import (
    DEFAULT_SCENES,
    V44Config,
    json_safe,
    load_baseline_bundle,
    read_split,
    run_scenes,
)


def _scenes(args: argparse.Namespace) -> list[str]:
    if args.scenes:
        return [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    if args.split:
        return read_split(Path(args.split))
    return list(DEFAULT_SCENES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v44 native typed mask assembly.")
    parser.add_argument("--scenes", default="", help="Comma-separated scene list. Defaults to v44 probe5 scenes.")
    parser.add_argument("--split", default="", help="Optional split file; ignored when --scenes is set.")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--feature-backend", default="rgb_stats")
    parser.add_argument(
        "--strategy",
        default="core_first",
        choices=[
            "core_first",
            "core_first_strict",
            "pseudo_core",
            "balanced_recall",
            "repair_purity",
            "repair_completeness",
            "shuffled_d4rt",
            "no_temporal",
            "mask_only",
        ],
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-mask-area", type=int, default=400)
    parser.add_argument("--min-visibility", type=float, default=0.50)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--max-tubes-per-window", type=int, default=1920)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=4401)
    parser.add_argument("--core-area-ratio", type=float, default=0.0040)
    parser.add_argument("--part-area-ratio", type=float, default=0.0015)
    parser.add_argument("--mixed-area-ratio", type=float, default=0.045)
    parser.add_argument("--mixed-variance-threshold", type=float, default=0.19)
    parser.add_argument("--mixed-boundary-threshold", type=float, default=0.18)
    parser.add_argument("--unknown-min-support", type=int, default=2)
    parser.add_argument("--link-max-rank-gap", type=int, default=2)
    parser.add_argument("--link-min-shared-tubes", type=int, default=2)
    parser.add_argument("--link-min-score", type=float, default=0.26)
    parser.add_argument("--link-min-score-recall", type=float, default=0.18)
    parser.add_argument("--absorb-min-score", type=float, default=0.44)
    parser.add_argument(
        "--v37-baseline",
        default="outputs/audit/v43_2_v37_parity_adapter/v37_4d_rerun_with_counts/4d_memory_decision.json",
    )
    parser.add_argument(
        "--v41-baseline",
        default="outputs/audit/v41_1_native_support_metrics_probe5/native_support_metrics_summary.json",
    )
    args = parser.parse_args()

    scenes = _scenes(args)
    config = V44Config(
        scannet_root=Path(args.scannet_root),
        cache_root=Path(args.cache_root),
        backbone=str(args.backbone),
        min_mask_area=int(args.min_mask_area),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        max_tubes_per_window=int(args.max_tubes_per_window),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        feature_backend=str(args.feature_backend),
        strategy=str(args.strategy),
        seed=int(args.seed),
        core_area_ratio=float(args.core_area_ratio),
        part_area_ratio=float(args.part_area_ratio),
        mixed_area_ratio=float(args.mixed_area_ratio),
        mixed_variance_threshold=float(args.mixed_variance_threshold),
        mixed_boundary_threshold=float(args.mixed_boundary_threshold),
        unknown_min_support=int(args.unknown_min_support),
        link_max_rank_gap=int(args.link_max_rank_gap),
        link_min_shared_tubes=int(args.link_min_shared_tubes),
        link_min_score=float(args.link_min_score),
        link_min_score_recall=float(args.link_min_score_recall),
        absorb_min_score=float(args.absorb_min_score),
    )
    baseline = load_baseline_bundle(Path(args.v37_baseline), Path(args.v41_baseline))
    summary = run_scenes(scenes, config, output_root=Path(args.output_root), baseline=baseline)
    print(json.dumps(json_safe({
        "output_root": str(Path(args.output_root)),
        "strategy": args.strategy,
        "aggregate_metrics": summary.get("aggregate_metrics", {}),
        "gate": summary.get("gate", {}),
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
