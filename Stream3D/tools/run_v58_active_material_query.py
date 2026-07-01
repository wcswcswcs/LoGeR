from __future__ import annotations

import argparse

from stream4d_native.v58_active_material_query import (
    V58ActiveMaterialQueryConfig,
    build_v58_active_material_query,
    write_v58_active_material_query,
    write_v58_active_material_query_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v58 Phase3 active material query.")
    parser.add_argument("--phase2-root", default="outputs/audit/v58_counterfactual_explanation_dino_full_repair6")
    parser.add_argument(
        "--reprojection-candidate-rows-path",
        default="outputs/audit/v54_reprojection_ledger_k0all_conflict_veto018_skip_repeated_sig_stride1_probe5_q4096_notopup_max4000_skip/candidate_rows.csv",
    )
    parser.add_argument(
        "--reprojection-ledger-rows-path",
        default="outputs/audit/v54_reprojection_ledger_k0all_conflict_veto018_skip_repeated_sig_stride1_probe5_q4096_notopup_max4000_skip/reprojection_ledger_rows.csv",
    )
    parser.add_argument("--output-root", default="outputs/audit/v58_active_material_query")
    parser.add_argument("--visualization-root", default="outputs/audit/v58_visualizations/active_query")
    parser.add_argument("--primary-variant", default="E6_counterfactual_semantic_material_underseg")
    parser.add_argument("--query-budget", type=int, default=128)
    parser.add_argument("--max-target-frames", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=58)
    parser.add_argument("--max-visual-queries", type=int, default=6)
    args = parser.parse_args()

    cfg = V58ActiveMaterialQueryConfig(
        phase2_root=args.phase2_root,
        reprojection_candidate_rows_path=args.reprojection_candidate_rows_path,
        reprojection_ledger_rows_path=args.reprojection_ledger_rows_path,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        primary_variant=args.primary_variant,
        query_budget=args.query_budget,
        max_target_frames=args.max_target_frames,
        random_seed=args.random_seed,
        max_visual_queries=args.max_visual_queries,
    )
    result = build_v58_active_material_query(cfg)
    paths = write_v58_active_material_query(result, args.output_root)
    visualizations = write_v58_active_material_query_visualizations(
        result,
        args.visualization_root,
        max_visual_queries=args.max_visual_queries,
    )
    summary = result["summary"]
    for name, path in paths.items():
        print(f"{name}: {path}")
    for path in visualizations:
        print(f"visualization: {path}")
    print(f"eligible_deferred_observation_count: {summary['eligible_deferred_observation_count']}")
    print(f"query_budget: {summary['query_budget']}")
    print(f"query_count: {summary['query_count']}")
    print(f"best_fixed_query_entropy_reduction: {summary['best_fixed_query_entropy_reduction']}")
    print(f"Q0_query_to_confirm_rate: {summary['Q0_query_to_confirm_rate']}")
    print(f"Q6_entropy_reduction: {summary['Q6_entropy_reduction']}")
    print(f"Q6_valid_material_evidence_rate: {summary['Q6_valid_material_evidence_rate']}")
    print(f"Q6_query_to_confirm_rate: {summary['Q6_query_to_confirm_rate']}")
    print(f"Q6_real_minus_shuffled_query_AUC: {summary['Q6_real_minus_shuffled_query_AUC']}")
    print(f"Q6_real_minus_no_temporal_query_AUC: {summary['Q6_real_minus_no_temporal_query_AUC']}")
    print(f"gate.pass: {summary['gate']['pass']}")


if __name__ == "__main__":
    main()
