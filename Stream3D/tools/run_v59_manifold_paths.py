from __future__ import annotations

import argparse

from stream4d_native.v59_manifold_paths import (
    V59PathConfig,
    build_v59_manifold_paths,
    write_v59_manifold_paths,
    write_v59_path_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v59 Phase2 manifold path diagnostics.")
    parser.add_argument("--graph-root", default="outputs/audit/v59_phase1_graph")
    parser.add_argument("--output-root", default="outputs/audit/v59_phase2_paths")
    parser.add_argument("--visualization-root", default="outputs/audit/v59_visualizations/phase2")
    parser.add_argument("--allow-semantic-only", action="store_true")
    parser.add_argument("--allow-exclusion-paths", action="store_true")
    parser.add_argument("--shortcut-multisignal-min", type=int, default=0)
    parser.add_argument("--deferred-shortcuts-remain-tentative", action="store_true")
    parser.add_argument("--shortcut-min-posterior", type=float, default=0.0)
    parser.add_argument("--shortcut-min-margin", type=float, default=0.0)
    parser.add_argument("--shortcut-reject-exclusion", action="store_true")
    args = parser.parse_args()

    cfg = V59PathConfig(
        graph_root=args.graph_root,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        require_material_path=not args.allow_semantic_only,
        reject_exclusion_paths=not args.allow_exclusion_paths,
        shortcut_multisignal_min=args.shortcut_multisignal_min,
        deferred_shortcuts_remain_tentative=args.deferred_shortcuts_remain_tentative,
        shortcut_min_posterior=args.shortcut_min_posterior,
        shortcut_min_margin=args.shortcut_min_margin,
        shortcut_reject_exclusion=args.shortcut_reject_exclusion,
    )
    result = build_v59_manifold_paths(cfg)
    outputs = write_v59_manifold_paths(result, args.output_root)
    visuals = write_v59_path_visualizations(result, args.visualization_root)
    summary = result["summary"]
    print(
        {
            "path_summary": outputs["path_summary"],
            "path_rows": outputs["path_rows"],
            "shortcut_rows": outputs["shortcut_rows"],
            "path_metric_rows": outputs["path_metric_rows"],
            "visualization_status": visuals["visualization_status"],
            "gate": summary["gate"],
            "accepted_path_count": summary["accepted_path_count"],
            "path_precision_diagnostic": summary["path_precision_diagnostic"],
            "part_to_core_path_precision": summary["part_to_core_path_precision"],
            "shortcut_quarantine_precision": summary["shortcut_quarantine_precision"],
            "same_category_metric_available": summary["same_category_metric_available"],
            "same_category_false_path_rate_proxy": summary["same_category_false_path_rate_proxy"],
            "semantic_pairwise_baseline_false_path_rate_proxy": summary[
                "semantic_pairwise_baseline_false_path_rate_proxy"
            ],
        }
    )


if __name__ == "__main__":
    main()
