from __future__ import annotations

import argparse

from stream4d_native.v63_query_policy import V63QueryPolicyConfig, build_v63_query_policy, write_v63_query_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v63 Phase 2 pre-D4RT query policy selection.")
    parser.add_argument("--query-candidate-rows", default="outputs/audit/v63_query_candidates/query_candidate_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v63_query_policy")
    parser.add_argument("--visualization-root", default="outputs/audit/v63_visualizations/query_policy")
    parser.add_argument("--query-budget", type=int, default=64)
    args = parser.parse_args()

    cfg = V63QueryPolicyConfig(
        query_candidate_rows=args.query_candidate_rows,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        query_budget=args.query_budget,
    )
    result = build_v63_query_policy(cfg)
    outputs = write_v63_query_policy(result, cfg)
    print(
        {
            "outputs": outputs,
            "gate": result["summary"]["gate"],
            "real_policy_type_counts": result["summary"]["real_policy_type_counts"],
            "real_policy_action_counts": result["summary"]["real_policy_action_counts"],
            "control_query_counts": result["summary"]["control_query_counts"],
            "method_status": result["summary"]["method_status"],
        }
    )


if __name__ == "__main__":
    main()
