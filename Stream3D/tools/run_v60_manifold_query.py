from __future__ import annotations

import argparse

from stream4d_native.v60_manifold_query import (
    V60QueryConfig,
    build_v60_manifold_query,
    write_v60_manifold_query,
    write_v60_query_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v60 Phase5 diagnostic manifold-aware query.")
    parser.add_argument("--output-root", default="outputs/audit/v60_manifold_query")
    parser.add_argument("--visualization-root", default="outputs/audit/v60_visualizations/manifold_query")
    parser.add_argument("--query-budget", type=int, default=128)
    args = parser.parse_args()
    cfg = V60QueryConfig(output_root=args.output_root, visualization_root=args.visualization_root, query_budget=args.query_budget)
    result = build_v60_manifold_query(cfg)
    outputs = write_v60_manifold_query(result, args.output_root)
    visuals = write_v60_query_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "query_summary": outputs["query_summary"],
            "query_rows": outputs["query_rows"],
            "material_evidence_rows": outputs["material_evidence_rows"],
            "visualization_status": visuals["visualization_status"],
            "diagnostic_only_bypass": summary["diagnostic_only_bypass"],
            "gate": summary["gate"],
            "query_count": summary["query_count"],
            "valid_material_evidence_rate": summary["valid_material_evidence_rate"],
            "query_to_confirm_rate": summary["query_to_confirm_rate"],
            "query_to_quarantine_rate": summary["query_to_quarantine_rate"],
            "state_entropy_reduction": summary["state_entropy_reduction"],
        }
    )


if __name__ == "__main__":
    main()
