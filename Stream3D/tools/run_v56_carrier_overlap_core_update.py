from __future__ import annotations

import argparse

from stream4d_native.v56_core_update_carrier_overlap import (
    build_v56_carrier_overlap_core_update,
    write_v56_carrier_overlap_core_update,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v56 C4b direct carrier-overlap core update.")
    parser.add_argument("--output-root", default="outputs/audit/v56_core_update_c4b_carrier_overlap")
    parser.add_argument("--component-min-shared-carrier-count", type=int, default=1)
    parser.add_argument("--component-min-carrier-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--objectlet-min-component-count", type=int, default=1)
    parser.add_argument("--objectlet-min-total-shared-carriers", type=int, default=1)
    parser.add_argument("--history-evidence-roles", default="bridge,update")
    parser.add_argument("--allow-unobserved-mask", action="store_true")
    parser.add_argument("--allow-nonvisible-uv", action="store_true")
    args = parser.parse_args()
    payload = build_v56_carrier_overlap_core_update(
        component_min_shared_carrier_count=args.component_min_shared_carrier_count,
        component_min_carrier_overlap_ratio=args.component_min_carrier_overlap_ratio,
        objectlet_min_component_count=args.objectlet_min_component_count,
        objectlet_min_total_shared_carriers=args.objectlet_min_total_shared_carriers,
        history_evidence_roles=tuple(role.strip() for role in args.history_evidence_roles.split(",") if role.strip()),
        require_visible_uv=not args.allow_nonvisible_uv,
        require_observed_mask=not args.allow_unobserved_mask,
    )
    write_v56_carrier_overlap_core_update(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/core_update_summary.json",
            "gate": summary["gate"],
            "history_with_native_carrier_count": summary["history_with_native_carrier_count"],
            "candidate_objectlet_count": summary["candidate_objectlet_count"],
            "candidate_component_with_carrier_count": summary["candidate_component_with_carrier_count"],
            "confirmed_update_count": summary["confirmed_update_count"],
            "confirmed_added_component_count": summary["confirmed_added_component_count"],
            "history_temporal_span_mean": summary["history_temporal_span_mean"],
            "history_ARI": summary["history_ARI"],
            "history_purity": summary["history_purity"],
            "history_completeness": summary["history_completeness"],
            "update_precision_diagnostic": summary["update_precision_diagnostic"],
            "real_minus_shuffled_ARI": summary["real_minus_shuffled_ARI"],
            "real_minus_no_temporal_ARI": summary["real_minus_no_temporal_ARI"],
        }
    )


if __name__ == "__main__":
    main()
