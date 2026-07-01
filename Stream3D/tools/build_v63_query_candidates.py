from __future__ import annotations

import argparse

from stream4d_native.v63_query_candidates import (
    V63QueryCandidateConfig,
    build_v63_query_candidates,
    write_v63_query_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v63 balanced query candidate protocol.")
    parser.add_argument("--v62-query-candidate-rows", default="outputs/audit/v62_active_query_refresh/query_candidate_rows.csv")
    parser.add_argument("--v62-novelty-material-rows", default="outputs/audit/v62_increment_attribution/novelty_material_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v63_query_candidates")
    parser.add_argument("--visualization-root", default="outputs/audit/v63_visualizations/query_candidates")
    parser.add_argument("--per-type-budget", type=int, default=64)
    parser.add_argument("--per-control-budget", type=int, default=64)
    args = parser.parse_args()

    cfg = V63QueryCandidateConfig(
        v62_query_candidate_rows=args.v62_query_candidate_rows,
        v62_novelty_material_rows=args.v62_novelty_material_rows,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        per_type_budget=args.per_type_budget,
        per_control_budget=args.per_control_budget,
    )
    result = build_v63_query_candidates(cfg)
    outputs = write_v63_query_candidates(result, cfg)
    print(
        {
            "outputs": outputs,
            "gate": result["summary"]["gate"],
            "method_candidate_type_counts": result["summary"]["method_candidate_type_counts"],
            "baseline_control_counts": result["summary"]["baseline_control_counts"],
            "method_status": result["summary"]["method_status"],
        }
    )


if __name__ == "__main__":
    main()
