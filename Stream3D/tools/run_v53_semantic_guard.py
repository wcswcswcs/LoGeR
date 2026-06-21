from __future__ import annotations

import argparse

from stream4d_native.v53_semantic_guard import build_semantic_guard, write_semantic_guard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 5 semantic/colorhist guard diagnostic.")
    parser.add_argument("--support-rows", default="outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv")
    parser.add_argument("--mask-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument(
        "--objectlet-summary",
        default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/local_objectlet_summary.json",
    )
    parser.add_argument(
        "--objectlet-rows",
        default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/objectlet_rows.csv",
    )
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--objectlet-variant", default=None)
    parser.add_argument("--output-root", default="outputs/audit/v53_semantic_guard")
    args = parser.parse_args()
    payload = build_semantic_guard(
        support_rows_path=args.support_rows,
        mask_table_path=args.mask_table,
        objectlet_summary_path=args.objectlet_summary,
        objectlet_rows_path=args.objectlet_rows,
        support_variant=args.support_variant,
        objectlet_variant=args.objectlet_variant,
    )
    write_semantic_guard(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/semantic_summary.json",
            "objectlet_variant": summary.get("objectlet_variant"),
            "dense_semantic_available": summary.get("dense_semantic_available"),
            "feature_backend": summary.get("feature_backend"),
            "semantic_claim_allowed": summary.get("semantic_claim_allowed"),
            "feature_success_rate": summary.get("feature_success_rate"),
            "component_feature_success_rate": summary.get("component_feature_success_rate"),
            "objectlet_feature_success_rate": summary.get("objectlet_feature_success_rate"),
            "semantic_contradiction_AUC_diagnostic": summary.get("semantic_contradiction_AUC_diagnostic"),
            "underseg_detection_AUC_diagnostic": summary.get("underseg_detection_AUC_diagnostic"),
            "semantic_guard_signal_pass": summary.get("semantic_guard_signal_pass"),
            "semantic_veto_candidate_pass": summary.get("semantic_veto_candidate_pass"),
            "semantic_guard_method_enabled": summary.get("semantic_guard_method_enabled"),
            "disable_reason": summary.get("disable_reason"),
        }
    )


if __name__ == "__main__":
    main()
