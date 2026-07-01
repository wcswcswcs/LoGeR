from __future__ import annotations

import argparse

from stream4d_native.v60_manifold_paths_v2 import (
    build_v60_manifold_paths_v2,
    write_v60_manifold_paths_v2,
    write_v60_path_v2_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v60 Phase2 manifold path v2 audit.")
    parser.add_argument("--output-root", default="outputs/audit/v60_manifold_paths_v2")
    parser.add_argument("--visualization-root", default="outputs/audit/v60_visualizations/manifold_paths_v2")
    args = parser.parse_args()
    result = build_v60_manifold_paths_v2()
    outputs = write_v60_manifold_paths_v2(result, args.output_root)
    visuals = write_v60_path_v2_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "path_summary": outputs["path_summary"],
            "path_rows": outputs["path_rows"],
            "shortcut_rows": outputs["shortcut_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "accepted_path_count": summary["accepted_path_count"],
            "path_precision_diagnostic": summary["path_precision_diagnostic"],
            "shortcut_quarantine_precision": summary["shortcut_quarantine_precision"],
            "same_category_calibrated_gate_pass": summary["same_category_calibrated_gate_pass"],
        }
    )


if __name__ == "__main__":
    main()
