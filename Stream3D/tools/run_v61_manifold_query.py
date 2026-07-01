from __future__ import annotations

import argparse

from stream4d_native.v61_manifold_query import (
    V61ManifoldQueryConfig,
    build_v61_manifold_query,
    write_v61_manifold_query,
    write_v61_query_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v61 Phase4 manifold-aware query.")
    parser.add_argument("--output-root", default="outputs/audit/v61_manifold_query")
    parser.add_argument("--visualization-root", default="outputs/audit/v61_visualizations/query")
    parser.add_argument("--query-budget", type=int, default=128)
    args = parser.parse_args()
    cfg = V61ManifoldQueryConfig(
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        query_budget=args.query_budget,
    )
    result = build_v61_manifold_query(cfg)
    outputs = write_v61_manifold_query(result, args.output_root)
    visuals = write_v61_query_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "query_summary": outputs["query_summary"],
            "query_rows": outputs["query_rows"],
            "material_evidence_rows": outputs["material_evidence_rows"],
            "query_metric_rows": outputs["query_metric_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "query_count": summary["query_count"],
            "candidate_pool_count": summary["candidate_pool_count"],
            "valid_material_evidence_rate": summary["valid_material_evidence_rate"],
            "query_to_confirm_or_quarantine_rate": summary["query_to_confirm_or_quarantine_rate"],
            "state_entropy_reduction": summary["state_entropy_reduction"],
            "real_minus_shuffled_query_AUC": summary["real_minus_shuffled_query_AUC"],
            "real_minus_no_temporal_query_AUC": summary["real_minus_no_temporal_query_AUC"],
        }
    )


if __name__ == "__main__":
    main()
