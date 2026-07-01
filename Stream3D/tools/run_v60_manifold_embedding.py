from __future__ import annotations

import argparse

from stream4d_native.v60_manifold_embedding import (
    V60EmbeddingConfig,
    build_v60_manifold_embedding,
    write_v60_embedding_visualizations,
    write_v60_manifold_embedding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v60 Phase3 global manifold embedding audit.")
    parser.add_argument("--output-root", default="outputs/audit/v60_manifold_embedding")
    parser.add_argument("--visualization-root", default="outputs/audit/v60_visualizations/manifold_embedding")
    parser.add_argument("--high-confidence-margin", type=float, default=0.60)
    args = parser.parse_args()
    cfg = V60EmbeddingConfig(
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        high_confidence_margin=args.high_confidence_margin,
    )
    result = build_v60_manifold_embedding(cfg)
    outputs = write_v60_manifold_embedding(result, args.output_root)
    visuals = write_v60_embedding_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "embedding_summary": outputs["embedding_summary"],
            "node_state_rows": outputs["node_state_rows"],
            "manifold_metric_rows": outputs["manifold_metric_rows"],
            "visualization_status": visuals["visualization_status"],
            "selected_variant": summary["selected_variant"],
            "gate": summary["gate"],
            "confirmed_node_count": summary["confirmed_node_count"],
            "tentative_node_count": summary["tentative_node_count"],
            "quarantine_node_count": summary["quarantine_node_count"],
            "core_purity": summary["core_purity"],
            "core_completeness": summary["core_completeness"],
            "expanded_completeness": summary["expanded_completeness"],
            "conflict_rate": summary["conflict_rate"],
        }
    )


if __name__ == "__main__":
    main()
