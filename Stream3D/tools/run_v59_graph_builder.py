from __future__ import annotations

import argparse

from stream4d_native.v59_graph_builder import (
    V59GraphBuilderConfig,
    build_v59_graph,
    write_v59_graph,
    write_v59_graph_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v59 Phase1 typed semantic-material graph builder.")
    parser.add_argument("--output-root", default="outputs/audit/v59_phase1_graph")
    parser.add_argument("--visualization-root", default="outputs/audit/v59_visualizations/phase1")
    parser.add_argument("--semantic-top-k-per-observation", type=int, default=3)
    parser.add_argument("--material-top-k-per-mask", type=int, default=5)
    parser.add_argument("--reprojection-max-edges-per-candidate", type=int, default=3)
    parser.add_argument("--semantic-min-score", type=float, default=0.35)
    args = parser.parse_args()

    cfg = V59GraphBuilderConfig(
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        semantic_top_k_per_observation=args.semantic_top_k_per_observation,
        material_top_k_per_mask=args.material_top_k_per_mask,
        reprojection_max_edges_per_candidate=args.reprojection_max_edges_per_candidate,
        semantic_min_score=args.semantic_min_score,
    )
    result = build_v59_graph(cfg)
    outputs = write_v59_graph(result, args.output_root)
    visuals = write_v59_graph_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "graph_summary": outputs["graph_summary"],
            "node_rows": outputs["node_rows"],
            "edge_rows": outputs["edge_rows"],
            "graph_invariant_rows": outputs["graph_invariant_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "node_count_by_type": summary["node_count_by_type"],
            "edge_count_by_type": summary["edge_count_by_type"],
            "history_manifold_count": summary["history_manifold_count"],
            "underseg_bridge_edge_count": summary["underseg_bridge_edge_count"],
            "same_frame_cannot_link_edge_count": summary["same_frame_cannot_link_edge_count"],
        }
    )


if __name__ == "__main__":
    main()
