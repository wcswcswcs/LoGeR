from __future__ import annotations

import argparse

from stream4d_native.v53_local_objectlets import build_local_objectlets, write_local_objectlets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 6 local objectlet diagnostic.")
    parser.add_argument("--support-rows", default="outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv")
    parser.add_argument("--candidate-rows", default="outputs/audit/v53_reprojection_ledger_conflict_veto/candidate_rows.csv")
    parser.add_argument("--ledger-rows", default="outputs/audit/v53_reprojection_ledger_conflict_veto/reprojection_ledger_rows.csv")
    parser.add_argument("--representative-rows", default="outputs/audit/v53_representative_observations_k8_underseg_cap_fixed/representative_mask_rows.csv")
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--representative-variant", default="K8_underseg_capped_partial_repair")
    parser.add_argument("--output-root", default="outputs/audit/v53_local_objectlets")
    args = parser.parse_args()
    payload = build_local_objectlets(
        support_rows_path=args.support_rows,
        candidate_rows_path=args.candidate_rows,
        ledger_rows_path=args.ledger_rows,
        representative_rows_path=args.representative_rows,
        support_variant=args.support_variant,
        representative_variant=args.representative_variant,
    )
    write_local_objectlets(args.output_root, payload)
    best = payload["summary"]["best_real_row"]
    print(
        {
            "summary": f"{args.output_root}/local_objectlet_summary.json",
            "best_real_variant": payload["summary"]["best_real_variant"],
            "any_relaxed_gate_pass": payload["summary"]["any_relaxed_gate_pass"],
            "any_success_gate_pass": payload["summary"]["any_success_gate_pass"],
            "best_4D_ARI": best.get("4D_ARI"),
            "best_4D_purity": best.get("4D_purity"),
            "best_4D_completeness": best.get("4D_completeness"),
            "best_conflict_rate": best.get("conflict_rate"),
            "best_outside_residual_mean": best.get("outside_residual_mean"),
        }
    )


if __name__ == "__main__":
    main()
