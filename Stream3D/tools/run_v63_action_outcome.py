from __future__ import annotations

import argparse

from stream4d_native.v63_action_outcome import V63ActionOutcomeConfig, build_v63_action_outcome, write_v63_action_outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v63 Phase 4 action outcome and utility evaluation.")
    parser.add_argument("--d4rt-query-result-rows", default="outputs/audit/v63_d4rt_query/query_result_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v63_action_outcome")
    parser.add_argument("--visualization-root", default="outputs/audit/v63_visualizations/action_outcome")
    parser.add_argument("--min-accepted-frames", type=int, default=2)
    parser.add_argument("--min-in-bounds-ratio", type=float, default=0.80)
    parser.add_argument("--min-source-score", type=float, default=0.25)
    args = parser.parse_args()

    cfg = V63ActionOutcomeConfig(
        d4rt_query_result_rows=args.d4rt_query_result_rows,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        min_accepted_frames=args.min_accepted_frames,
        min_in_bounds_ratio=args.min_in_bounds_ratio,
        min_source_score=args.min_source_score,
    )
    result = build_v63_action_outcome(cfg)
    outputs = write_v63_action_outcome(result, cfg)
    print(
        {
            "outputs": outputs,
            "gate": result["summary"]["gate"],
            "real_minus_best_fixed_utility": result["summary"]["real_minus_best_fixed_utility"],
            "real_minus_shuffled_query_utility": result["summary"]["real_minus_shuffled_query_utility"],
            "real_minus_no_temporal_query_utility": result["summary"]["real_minus_no_temporal_query_utility"],
            "method_status": result["summary"]["method_status"],
        }
    )


if __name__ == "__main__":
    main()
