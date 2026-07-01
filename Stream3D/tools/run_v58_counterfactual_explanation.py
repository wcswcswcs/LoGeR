from __future__ import annotations

import argparse

from stream4d_native.v58_counterfactual_explanation import (
    V58CounterfactualConfig,
    build_v58_counterfactual_explanation,
    write_v58_counterfactual_explanation,
    write_v58_counterfactual_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v58 Phase2 counterfactual explanation.")
    parser.add_argument("--semantic-root", default="outputs/audit/v58_semantic_memory_dino_full_repair2")
    parser.add_argument("--support-rows-path", default="outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv")
    parser.add_argument("--history-rows-path", default="outputs/audit/v55_history_update/history_rows.csv")
    parser.add_argument("--history-update-rows-path", default="outputs/audit/v55_history_update/history_update_rows.csv")
    parser.add_argument(
        "--objectlet-rows-path",
        default="outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    )
    parser.add_argument("--v56-core-summary-path", default="outputs/audit/v56_core_update/core_update_summary.json")
    parser.add_argument("--v56-tentative-summary-path", default="outputs/audit/v56_tentative_support/tentative_support_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v58_counterfactual_explanation")
    parser.add_argument("--visualization-root", default="outputs/audit/v58_visualizations/explanation")
    parser.add_argument("--support-variant", default="I0_visible_tau0.10")
    parser.add_argument("--objectlet-underseg-variant", default="L11_dynamic_uncovered_gain_dup010")
    parser.add_argument("--primary-variant", default="E6_counterfactual_semantic_material_underseg")
    parser.add_argument("--k-sem", type=int, default=5)
    parser.add_argument("--k-mat", type=int, default=5)
    parser.add_argument("--max-modes", type=int, default=4)
    parser.add_argument("--max-observations", type=int, default=None)
    args = parser.parse_args()

    cfg = V58CounterfactualConfig(
        semantic_root=args.semantic_root,
        support_rows_path=args.support_rows_path,
        history_rows_path=args.history_rows_path,
        history_update_rows_path=args.history_update_rows_path,
        objectlet_rows_path=args.objectlet_rows_path,
        v56_core_summary_path=args.v56_core_summary_path,
        v56_tentative_summary_path=args.v56_tentative_summary_path,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        support_variant=args.support_variant,
        objectlet_underseg_variant=args.objectlet_underseg_variant,
        primary_variant=args.primary_variant,
        k_sem=args.k_sem,
        k_mat=args.k_mat,
        max_modes=args.max_modes,
        max_observations=args.max_observations,
    )
    result = build_v58_counterfactual_explanation(cfg)
    paths = write_v58_counterfactual_explanation(result, args.output_root)
    paths["visualization"] = write_v58_counterfactual_visualization(result, args.visualization_root, tag=args.primary_variant)
    summary = result["summary"]
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"primary_variant: {summary['primary_variant']}")
    print(f"observation_count: {summary['observation_count']}")
    print(f"explanation_candidate_count: {summary['explanation_candidate_count']}")
    print(f"actionable_count: {summary['actionable_count']}")
    print(f"deferred_count: {summary['deferred_count']}")
    print(f"assign_precision_diagnostic: {summary['assign_precision_diagnostic']}")
    print(f"partial_precision_diagnostic: {summary['partial_precision_diagnostic']}")
    print(f"underseg_precision_diagnostic: {summary['underseg_precision_diagnostic']}")
    print(f"new_birth_precision_diagnostic: {summary['new_birth_precision_diagnostic']}")
    print(f"false_history_update_rate: {summary['false_history_update_rate']}")
    print(f"gate.pass: {summary['gate']['pass']}")


if __name__ == "__main__":
    main()
