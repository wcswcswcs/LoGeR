from __future__ import annotations

import argparse

from stream4d_native.v61_graph_v3 import V61GraphV3Config, build_v61_graph_v3, write_v61_graph_v3, write_v61_graph_v3_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v61 Phase1 graph v3 material candidate generation.")
    parser.add_argument("--output-root", default="outputs/audit/v61_graph_v3")
    parser.add_argument("--visualization-root", default="outputs/audit/v61_visualizations/graph_v3")
    parser.add_argument("--semantic-topk-per-observation", type=int, default=5)
    parser.add_argument("--max-candidates-per-material", type=int, default=8)
    args = parser.parse_args()
    cfg = V61GraphV3Config(
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        semantic_topk_per_observation=args.semantic_topk_per_observation,
        max_candidates_per_material=args.max_candidates_per_material,
    )
    result = build_v61_graph_v3(cfg)
    outputs = write_v61_graph_v3(result, args.output_root)
    visuals = write_v61_graph_v3_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "graph_v3_summary": outputs["graph_v3_summary"],
            "material_candidate_rows": outputs["material_candidate_rows"],
            "edge_rows": outputs["edge_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "material_node_count": summary["material_node_count"],
            "material_candidate_pair_count": summary["material_candidate_pair_count"],
            "material_nodes_with_candidate_rate": summary["material_nodes_with_candidate_rate"],
            "candidate_recall_at_5": summary["candidate_recall_at_5"],
            "same_category_candidate_confusion_rate": summary["same_category_candidate_confusion_rate"],
            "semantic_only_candidate_confusion_rate": summary["semantic_only_candidate_confusion_rate"],
        }
    )


if __name__ == "__main__":
    main()
