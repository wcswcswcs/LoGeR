from __future__ import annotations

import argparse

from stream4d_native.v53_representative_observations import (
    build_representative_observations,
    write_representative_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 3 representative observation selection.")
    parser.add_argument("--support-rows", default="outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv")
    parser.add_argument("--mask-summary", default="outputs/audit/v53_mask_component_support_tau005/mask_summary_rows.csv")
    parser.add_argument("--chunk-component-rows", default="outputs/audit/v53_chunk_universe/chunk_component_rows.csv")
    parser.add_argument("--chunk-mask-rows", default="outputs/audit/v53_chunk_universe/chunk_mask_rows.csv")
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--gate-variant", default="K5_coverage_redundancy_cannot_link")
    parser.add_argument("--max-selected-ratio", type=float, default=0.60)
    parser.add_argument("--output-root", default="outputs/audit/v53_representative_observations")
    parser.add_argument("--visualization-root", default="outputs/audit/v53_visualizations/local_objectlets")
    args = parser.parse_args()
    payload = build_representative_observations(
        support_rows_path=args.support_rows,
        mask_summary_path=args.mask_summary,
        chunk_component_rows_path=args.chunk_component_rows,
        chunk_mask_rows_path=args.chunk_mask_rows,
        support_variant=args.support_variant,
        max_selected_ratio=args.max_selected_ratio,
        gate_variant=args.gate_variant,
    )
    write_representative_observations(args.output_root, payload, visualization_root=args.visualization_root)
    gate_row = next(row for row in payload["summary"]["variant_summaries"] if row["variant"] == args.gate_variant)
    print(
        {
            "summary": f"{args.output_root}/representative_summary.json",
            "gate": payload["summary"]["gate"],
            "representative_mode": payload["summary"]["representative_mode"],
            "gate_variant": args.gate_variant,
            "selected_observation_ratio": gate_row["selected_observation_ratio"],
            "component_coverage": gate_row["component_coverage"],
            "same_frame_conflict_rate": gate_row["same_frame_conflict_rate"],
            "underseg_selected_rate": gate_row["underseg_selected_rate"],
            "raw_underseg_rate": gate_row["raw_underseg_rate"],
        }
    )


if __name__ == "__main__":
    main()
