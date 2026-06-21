from __future__ import annotations

import argparse

from stream4d_native.v53_reprojection_ledger import build_reprojection_ledger, write_reprojection_ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 4 reprojection explanation ledger.")
    parser.add_argument("--carrier-table", default="outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv")
    parser.add_argument("--mask-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument("--support-rows", default="outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv")
    parser.add_argument("--representative-rows", default="outputs/audit/v53_representative_observations_k8_underseg_cap_fixed/representative_mask_rows.csv")
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--representative-variant", default="K8_underseg_capped_partial_repair")
    parser.add_argument("--max-candidates", type=int, default=240)
    parser.add_argument("--min-visible-carriers", type=int, default=3)
    parser.add_argument("--max-components-per-candidate", type=int, default=0)
    parser.add_argument("--skip-no-related-measurement", action="store_true")
    parser.add_argument("--include-repeated-support-candidates", action="store_true")
    parser.add_argument("--repeated-support-min-frames", type=int, default=4)
    parser.add_argument("--repeated-support-min-components", type=int, default=2)
    parser.add_argument("--repeated-support-min-w-visible", type=float, default=0.50)
    parser.add_argument("--repeated-support-max-components", type=int, default=128)
    parser.add_argument("--repeated-support-max-groups-per-scene", type=int, default=80)
    parser.add_argument(
        "--max-candidate-conflict-rate",
        type=float,
        default=-1.0,
        help="If >=0, apply a candidate-level same-frame conflict hard veto before summarizing the ledger.",
    )
    parser.add_argument("--output-root", default="outputs/audit/v53_reprojection_ledger")
    parser.add_argument("--visualization-root", default="outputs/audit/v53_visualizations/reprojection")
    args = parser.parse_args()
    payload = build_reprojection_ledger(
        carrier_table_path=args.carrier_table,
        mask_table_path=args.mask_table,
        support_rows_path=args.support_rows,
        representative_rows_path=args.representative_rows,
        support_variant=args.support_variant,
        representative_variant=args.representative_variant,
        max_candidates=args.max_candidates,
        min_visible_carriers=args.min_visible_carriers,
        max_components_per_candidate=args.max_components_per_candidate,
        skip_no_related_measurement=bool(args.skip_no_related_measurement),
        max_candidate_conflict_rate=args.max_candidate_conflict_rate if args.max_candidate_conflict_rate >= 0.0 else None,
        include_repeated_support_candidates=bool(args.include_repeated_support_candidates),
        repeated_support_min_frames=args.repeated_support_min_frames,
        repeated_support_min_components=args.repeated_support_min_components,
        repeated_support_min_w_visible=args.repeated_support_min_w_visible,
        repeated_support_max_components=args.repeated_support_max_components,
        repeated_support_max_groups_per_scene=args.repeated_support_max_groups_per_scene,
    )
    write_reprojection_ledger(args.output_root, payload, visualization_root=args.visualization_root)
    print(
        {
            "summary": f"{args.output_root}/reprojection_summary.json",
            "gate": payload["summary"]["gate"],
            "candidate_count": payload["summary"]["candidate_count"],
            "ledger_row_count": payload["summary"]["ledger_row_count"],
            "reprojection_success_rate": payload["summary"]["reprojection_success_rate"],
            "outside_all_related_masks_ratio_mean": payload["summary"]["outside_all_related_masks_ratio_mean"],
            "same_frame_exclusion_violation_rate": payload["summary"]["same_frame_exclusion_violation_rate"],
            "same_gt_precision": payload["summary"]["reprojection_success_same_GT_precision"],
            "repeated_support_candidate_count": payload["summary"]["repeated_support_candidate_count"],
        }
    )


if __name__ == "__main__":
    main()
