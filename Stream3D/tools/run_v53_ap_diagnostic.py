from __future__ import annotations

import argparse

from stream4d_native.v53_ap_diagnostic import build_v53_ap_diagnostic, write_v53_ap_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 11 AP smoke/diagnostic.")
    parser.add_argument("--objectlet-summary", default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/local_objectlet_summary.json")
    parser.add_argument("--objectlet-rows", default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/objectlet_rows.csv")
    parser.add_argument("--mask-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument("--v53-native-carrier-summary", default="outputs/audit/v53_native_carrier_materialization/native_carrier_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v53_ap_diagnostic")
    parser.add_argument("--output-config-prefix", default="v53_local_objectlets_l6_best_legal")
    parser.add_argument("--export-mask-sample-stride", type=int, default=4)
    parser.add_argument("--export-mask-max-pixels", type=int, default=30000)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    args = parser.parse_args()
    payload = build_v53_ap_diagnostic(
        objectlet_summary_path=args.objectlet_summary,
        objectlet_rows_path=args.objectlet_rows,
        mask_table_path=args.mask_table,
        v53_native_carrier_summary_path=args.v53_native_carrier_summary,
        output_root=args.output_root,
        output_config_prefix=args.output_config_prefix,
        export_mask_sample_stride=args.export_mask_sample_stride,
        export_mask_max_pixels=args.export_mask_max_pixels,
        export_nn_radius=args.export_nn_radius,
    )
    write_v53_ap_diagnostic(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/ap_export_summary.json",
            "gate": payload["gate"],
            "best_AP": payload["summary"]["best_AP"],
            "best_AP50": payload["summary"]["best_AP50"],
            "best_AP25": payload["summary"]["best_AP25"],
            "best_AP_variant": payload["summary"]["best_AP_variant"],
            "method_safe_ap_available": payload["gate"]["method_safe_ap_available"],
        }
    )


if __name__ == "__main__":
    main()
