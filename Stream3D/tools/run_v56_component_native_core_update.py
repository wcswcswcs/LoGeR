from __future__ import annotations

import argparse

from stream4d_native.v56_core_update_component_native import (
    build_v56_component_native_core_update,
    write_v56_component_native_core_update,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v56 C4 component-level native core update.")
    parser.add_argument("--output-root", default="outputs/audit/v56_core_update_c4_component_native")
    parser.add_argument("--component-min-shared-support", type=int, default=5)
    parser.add_argument("--component-min-shared-ratio", type=float, default=0.30)
    parser.add_argument("--objectlet-min-component-count", type=int, default=2)
    parser.add_argument("--objectlet-min-total-shared-support", type=int, default=20)
    parser.add_argument("--history-evidence-roles", default="bridge,update")
    args = parser.parse_args()
    payload = build_v56_component_native_core_update(
        component_min_shared_support=args.component_min_shared_support,
        component_min_shared_ratio=args.component_min_shared_ratio,
        objectlet_min_component_count=args.objectlet_min_component_count,
        objectlet_min_total_shared_support=args.objectlet_min_total_shared_support,
        history_evidence_roles=tuple(role.strip() for role in args.history_evidence_roles.split(",") if role.strip()),
    )
    write_v56_component_native_core_update(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/core_update_summary.json",
            "gate": summary["gate"],
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

