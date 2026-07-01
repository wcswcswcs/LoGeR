from __future__ import annotations

import argparse

from stream4d_native.v61_global_embedding import (
    build_v61_global_embedding,
    write_v61_global_embedding,
    write_v61_global_embedding_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v61 Phase2 global material ownership embedding.")
    parser.add_argument("--output-root", default="outputs/audit/v61_global_embedding")
    parser.add_argument("--visualization-root", default="outputs/audit/v61_visualizations/global_embedding")
    args = parser.parse_args()
    result = build_v61_global_embedding()
    outputs = write_v61_global_embedding(result, args.output_root)
    visuals = write_v61_global_embedding_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "embedding_summary": outputs["embedding_summary"],
            "material_state_rows": outputs["material_state_rows"],
            "observation_explanation_rows": outputs["observation_explanation_rows"],
            "energy_rows": outputs["energy_rows"],
            "visualization_status": visuals["visualization_status"],
            "selected_variant": summary["selected_variant"],
            "gate": summary["gate"],
            "confirmed_material_count": summary["confirmed_material_count"],
            "tentative_material_count": summary["tentative_material_count"],
            "shared_material_count": summary["shared_material_count"],
            "unknown_material_count": summary["unknown_material_count"],
            "core_purity": summary["core_purity"],
            "core_completeness": summary["core_completeness"],
            "expanded_completeness": summary["expanded_completeness"],
        }
    )


if __name__ == "__main__":
    main()
