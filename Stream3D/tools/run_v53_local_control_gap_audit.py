from __future__ import annotations

import argparse

from stream4d_native.v53_local_control_gap_audit import (
    build_local_control_gap_audit,
    write_local_control_gap_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 local D4RT-vs-mask-only control gap audit.")
    parser.add_argument("--support-rows", default="outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv")
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument(
        "--local-summary",
        default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000_l11_dynamic/local_objectlet_summary.json",
    )
    parser.add_argument(
        "--objectlet-rows",
        default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000_l11_dynamic/objectlet_rows.csv",
    )
    parser.add_argument("--output-root", default="outputs/audit/v53_local_control_gap_audit")
    args = parser.parse_args()
    payload = build_local_control_gap_audit(
        support_rows_path=args.support_rows,
        support_variant=args.support_variant,
        local_summary_path=args.local_summary,
        objectlet_rows_path=args.objectlet_rows,
    )
    write_local_control_gap_audit(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/local_control_gap_summary.json",
            "best_method_variant": summary["best_method_variant"],
            "best_method_real_minus_mask_only_ARI": summary["best_method_real_minus_mask_only_ARI"],
            "best_l11_variant": summary["best_l11_variant"],
            "best_l11_real_minus_mask_only_ARI": summary["best_l11_real_minus_mask_only_ARI"],
            "repair_conclusion": summary["repair_conclusion"],
        }
    )


if __name__ == "__main__":
    main()
