from __future__ import annotations

import argparse

from stream4d_native.v60_manifold_refinement import (
    build_v60_manifold_refinement,
    write_v60_manifold_refinement,
    write_v60_refinement_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v60 Phase4 diagnostic manifold refinement.")
    parser.add_argument("--output-root", default="outputs/audit/v60_manifold_refinement")
    parser.add_argument("--visualization-root", default="outputs/audit/v60_visualizations/manifold_refinement")
    args = parser.parse_args()
    result = build_v60_manifold_refinement()
    outputs = write_v60_manifold_refinement(result, args.output_root)
    visuals = write_v60_refinement_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "refinement_summary": outputs["refinement_summary"],
            "refinement_rows": outputs["refinement_rows"],
            "visualization_status": visuals["visualization_status"],
            "diagnostic_only_bypass": summary["diagnostic_only_bypass"],
            "gate": summary["gate"],
            "promoted_node_count": summary["promoted_node_count"],
            "quarantine_precision_diagnostic": summary["quarantine_precision_diagnostic"],
            "false_promotion_rate": summary["false_promotion_rate"],
        }
    )


if __name__ == "__main__":
    main()
