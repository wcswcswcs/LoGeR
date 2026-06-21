from __future__ import annotations

import argparse

from stream4d_native.v53_mask_component_support import build_mask_component_support, write_mask_component_support


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 1 CropFormer mask-to-component support audit.")
    parser.add_argument("--carrier-table", default="outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv")
    parser.add_argument("--mask-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument("--max-union-unique-carriers", type=int, default=32)
    parser.add_argument("--min-visibility-prob", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--extra-visible-tau", type=float, action="append", default=[])
    parser.add_argument("--gate-variant", default="I0_visible_tau0.10")
    parser.add_argument("--output-root", default="outputs/audit/v53_mask_component_support")
    parser.add_argument("--visualization-root", default="outputs/audit/v53_visualizations/local_objectlets")
    args = parser.parse_args()
    payload = build_mask_component_support(
        carrier_table_path=args.carrier_table,
        mask_table_path=args.mask_table,
        max_union_unique_carriers=args.max_union_unique_carriers,
        min_visibility_prob=args.min_visibility_prob,
        min_confidence=args.min_confidence,
        extra_visible_taus=list(args.extra_visible_tau),
        gate_variant=str(args.gate_variant),
    )
    write_mask_component_support(args.output_root, payload, visualization_root=args.visualization_root)
    main_summary = payload["summary"]["main_summary"]
    print(
        {
            "summary": f"{args.output_root}/support_summary.json",
            "gate": payload["summary"]["gate"],
            "mask_count": main_summary["mask_count"],
            "component_count": main_summary["component_count"],
            "incidence_row_count": main_summary["incidence_row_count"],
            "zero_component_mask_ratio": main_summary["zero_component_mask_ratio"],
        }
    )


if __name__ == "__main__":
    main()
