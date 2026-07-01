from __future__ import annotations

import argparse

from stream4d_native.v62_native_field import V62NativeFieldConfig, build_v62_native_field, write_v62_native_field, write_v62_native_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 7 native component field export.")
    parser.add_argument("--material-candidate-rows", default="outputs/audit/v61_graph_v3/material_candidate_rows.csv")
    parser.add_argument("--v56-core-summary", default="outputs/audit/v56_core_update/core_update_summary.json")
    parser.add_argument("--v56-tentative-summary", default="outputs/audit/v56_tentative_support/tentative_support_summary.json")
    parser.add_argument("--v61-embedding-summary", default="outputs/audit/v61_global_embedding/embedding_summary.json")
    parser.add_argument("--v61-native-summary", default="outputs/audit/v61_native_field/native_field_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v62_native_field")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/native_field")
    args = parser.parse_args()
    cfg = V62NativeFieldConfig(
        v61_native_summary_path=args.v61_native_summary,
        material_candidate_rows_path=args.material_candidate_rows,
        v56_core_summary_path=args.v56_core_summary,
        v56_tentative_summary_path=args.v56_tentative_summary,
        v61_embedding_summary_path=args.v61_embedding_summary,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
    )
    result = build_v62_native_field(cfg)
    outputs = write_v62_native_field(result, args.output_root)
    visuals = write_v62_native_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"]})


if __name__ == "__main__":
    main()
