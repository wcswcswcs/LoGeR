from __future__ import annotations

import argparse

from stream4d_native.v60_graph_v2 import build_v60_graph_v2, write_v60_graph_v2, write_v60_graph_v2_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v60 Phase1 graph v2 construction.")
    parser.add_argument("--output-root", default="outputs/audit/v60_graph_v2")
    parser.add_argument("--visualization-root", default="outputs/audit/v60_visualizations/graph_v2")
    args = parser.parse_args()
    result = build_v60_graph_v2()
    outputs = write_v60_graph_v2(result, args.output_root)
    visuals = write_v60_graph_v2_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "graph_summary": outputs["graph_summary"],
            "node_rows": outputs["node_rows"],
            "edge_rows": outputs["edge_rows"],
            "edge_cost_rows": outputs["edge_cost_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "edge_cost_nonempty_rate_by_type": summary["edge_cost_nonempty_rate_by_type"],
            "shortcut_edge_count": summary["shortcut_edge_count"],
            "hard_constraint_violation_count": summary["hard_constraint_violation_count"],
        }
    )


if __name__ == "__main__":
    main()
