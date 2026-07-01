from __future__ import annotations

import argparse

from stream4d_native.v61_refinement import (
    V61RefinementConfig,
    build_v61_refinement,
    write_v61_refinement,
    write_v61_refinement_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v61 Phase3 manifold refinement.")
    parser.add_argument("--output-root", default="outputs/audit/v61_refinement")
    parser.add_argument("--visualization-root", default="outputs/audit/v61_visualizations/refinement")
    args = parser.parse_args()
    cfg = V61RefinementConfig(output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v61_refinement(cfg)
    outputs = write_v61_refinement(result, args.output_root)
    visuals = write_v61_refinement_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "refinement_summary": outputs["refinement_summary"],
            "refinement_rows": outputs["refinement_rows"],
            "material_state_after_refinement": outputs["material_state_after_refinement"],
            "visualization_status": visuals["visualization_status"],
            "selected_variant": summary["selected_variant"],
            "gate": summary["gate"],
            "promoted_node_count": summary["promoted_node_count"],
            "quarantined_node_count": summary["quarantined_node_count"],
            "quarantine_precision_diagnostic": summary["quarantine_precision_diagnostic"],
            "core_purity_gain": summary["core_purity_gain"],
            "core_completeness_gain": summary["core_completeness_gain"],
            "expanded_completeness_drop": summary["expanded_completeness_drop"],
        }
    )


if __name__ == "__main__":
    main()
