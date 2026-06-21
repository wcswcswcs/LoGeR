from __future__ import annotations

import argparse

from stream4d_native.v53_chunk_universe import build_chunk_universe, write_chunk_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 2 chunk universe audit.")
    parser.add_argument("--carrier-table", default="outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv")
    parser.add_argument("--mask-table", default="outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv")
    parser.add_argument("--max-union-unique-carriers", type=int, default=32)
    parser.add_argument("--min-visibility-prob", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-stride", type=int, default=16)
    parser.add_argument("--output-root", default="outputs/audit/v53_chunk_universe")
    parser.add_argument("--visualization-root", default="outputs/audit/v53_visualizations/local_objectlets")
    args = parser.parse_args()
    payload = build_chunk_universe(
        carrier_table_path=args.carrier_table,
        mask_table_path=args.mask_table,
        max_union_unique_carriers=args.max_union_unique_carriers,
        min_visibility_prob=args.min_visibility_prob,
        min_confidence=args.min_confidence,
        chunk_size=args.chunk_size,
        chunk_stride=args.chunk_stride,
    )
    write_chunk_universe(args.output_root, payload, visualization_root=args.visualization_root)
    print(
        {
            "summary": f"{args.output_root}/chunk_summary.json",
            "gate": payload["summary"]["gate"],
            "chunk_count": payload["summary"]["chunk_count"],
            "components_per_chunk_mean": payload["summary"]["components_per_chunk_mean"],
            "chunk_component_coverage": payload["summary"]["chunk_component_coverage"],
            "component_visibility_frame_count_mean": payload["summary"]["component_visibility_frame_count_mean"],
            "weak_scale_chunk_count": payload["summary"]["weak_scale_chunk_count"],
        }
    )


if __name__ == "__main__":
    main()
