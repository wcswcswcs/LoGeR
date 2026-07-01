from __future__ import annotations

import argparse
import json

from stream4d_native.v58_semantic_memory import (
    V58SemanticMemoryConfig,
    build_v58_semantic_memory,
    write_v58_semantic_memory_visualization,
    write_v58_semantic_memory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v58 SOMA-4D semantic feature bank and memory initialization.")
    parser.add_argument("--output-root", default="outputs/audit/v58_semantic_memory")
    parser.add_argument("--visualization-root", default="outputs/audit/v58_visualizations/semantic_memory")
    parser.add_argument("--support-rows-path", default="outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv")
    parser.add_argument("--history-rows-path", default="outputs/audit/v55_history_update/history_rows.csv")
    parser.add_argument("--history-update-rows-path", default="outputs/audit/v55_history_update/history_update_rows.csv")
    parser.add_argument(
        "--objectlet-rows-path",
        default="outputs/audit/v54_local_objectlets_k0all_conflict_veto018_skip_repeated_sig_l12_stride1_probe5_q4096_notopup_max4000_skip/objectlet_rows.csv",
    )
    parser.add_argument("--support-variant", default="I0_visible_tau0.10")
    parser.add_argument("--objectlet-underseg-variant", default="L11_dynamic_uncovered_gain_dup010")
    parser.add_argument("--backend", choices=["dinov2_timm", "colorhist"], default="dinov2_timm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--short-side", type=int, default=518)
    parser.add_argument("--max-modes", type=int, default=4)
    parser.add_argument("--max-mask-observations", type=int, default=None)
    parser.add_argument("--no-mask-feature-vectors", action="store_true")
    args = parser.parse_args()

    config = V58SemanticMemoryConfig(
        support_rows_path=args.support_rows_path,
        history_rows_path=args.history_rows_path,
        history_update_rows_path=args.history_update_rows_path,
        objectlet_rows_path=args.objectlet_rows_path,
        output_root=args.output_root,
        visualization_root=args.visualization_root,
        support_variant=args.support_variant,
        objectlet_underseg_variant=args.objectlet_underseg_variant,
        backend=args.backend,
        device=args.device,
        checkpoint=args.checkpoint,
        short_side=int(args.short_side),
        max_modes=int(args.max_modes),
        max_mask_observations=args.max_mask_observations,
        write_mask_feature_vectors=not bool(args.no_mask_feature_vectors),
    )
    result = build_v58_semantic_memory(config)
    paths = write_v58_semantic_memory(result, args.output_root)
    paths["visualization"] = write_v58_semantic_memory_visualization(
        result,
        args.visualization_root,
        tag=str(args.backend).replace("/", "_"),
    )
    print(
        json.dumps(
            {
                "paths": paths,
                "gate": result["summary"].get("gate"),
                "semantic_claim_allowed": result["summary"].get("semantic_claim_allowed"),
                "feature_success_rate": result["summary"].get("feature_success_rate"),
                "history_shortlist_recall@3": result["summary"].get("history_shortlist_recall@3"),
                "underseg_detection_AUC": result["summary"].get("underseg_detection_AUC"),
                "semantic_claim_blockers": result["summary"].get("semantic_claim_blockers"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
