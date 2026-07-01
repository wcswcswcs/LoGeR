from __future__ import annotations

import argparse

from stream4d_native.v62_solver_v2 import V62SolverV2Config, build_v62_solver_v2, write_v62_solver_v2, write_v62_solver_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 3 SOMA-Manifold solver v2.")
    parser.add_argument("--material-candidate-rows", default="outputs/audit/v61_graph_v3/material_candidate_rows.csv")
    parser.add_argument("--v56-core-summary", default="outputs/audit/v56_core_update/core_update_summary.json")
    parser.add_argument("--v56-tentative-summary", default="outputs/audit/v56_tentative_support/tentative_support_summary.json")
    parser.add_argument("--v61-embedding-summary", default="outputs/audit/v61_global_embedding/embedding_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v62_solver_v2")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/solver_v2")
    args = parser.parse_args()
    cfg = V62SolverV2Config(
        material_candidate_rows_path=args.material_candidate_rows,
        v56_core_summary_path=args.v56_core_summary,
        v56_tentative_summary_path=args.v56_tentative_summary,
        v61_embedding_summary_path=args.v61_embedding_summary,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
    )
    result = build_v62_solver_v2(cfg)
    outputs = write_v62_solver_v2(result, args.output_root)
    visuals = write_v62_solver_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"]})


if __name__ == "__main__":
    main()
