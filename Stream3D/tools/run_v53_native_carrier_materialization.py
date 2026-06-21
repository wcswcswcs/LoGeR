from __future__ import annotations

import argparse

from stream4d_native.v53_native_carrier_materialization import (
    build_native_carrier_materialization,
    write_native_carrier_materialization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize v53 local objectlets to method-safe D4RT carrier support.")
    parser.add_argument("--carrier-table", default="outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv")
    parser.add_argument("--mask-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument(
        "--objectlet-summary",
        default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/local_objectlet_summary.json",
    )
    parser.add_argument(
        "--objectlet-rows",
        default="outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/objectlet_rows.csv",
    )
    parser.add_argument("--objectlet-variant", default=None)
    parser.add_argument("--output-root", default="outputs/audit/v53_native_carrier_materialization")
    parser.add_argument("--max-union-unique-carriers", type=int, default=32)
    parser.add_argument("--min-visibility-prob", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    payload = build_native_carrier_materialization(
        carrier_table_path=args.carrier_table,
        mask_table_path=args.mask_table,
        objectlet_summary_path=args.objectlet_summary,
        objectlet_rows_path=args.objectlet_rows,
        objectlet_variant=args.objectlet_variant,
        max_union_unique_carriers=args.max_union_unique_carriers,
        min_visibility_prob=args.min_visibility_prob,
        min_confidence=args.min_confidence,
    )
    write_native_carrier_materialization(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/native_carrier_summary.json",
            "variant": summary.get("objectlet_variant"),
            "native_carrier_materialization_pass": summary.get("native_carrier_materialization_pass"),
            "selected_objectlet_count": summary.get("selected_objectlet_count"),
            "selected_component_count": summary.get("selected_component_count"),
            "native_observation_row_count": summary.get("native_observation_row_count"),
            "native_unique_carrier_count": summary.get("native_unique_carrier_count"),
            "method_safe_ap_available": summary.get("method_safe_ap_available"),
            "AP_bridge_status": summary.get("AP_bridge_status"),
        }
    )


if __name__ == "__main__":
    main()
