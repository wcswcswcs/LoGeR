from __future__ import annotations

import argparse

from stream4d_native.v55_history_update import build_v55_history_update, write_v55_history_update


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v55 Phase 4 history-conditioned update.")
    parser.add_argument("--output-root", default="outputs/audit/v55_history_update")
    parser.add_argument("--visualization-root", default="outputs/audit/v55_visualizations/history_update")
    parser.add_argument("--history-evidence-roles", default="bridge,update")
    parser.add_argument("--cosupport-seed-ratio-min", type=float, default=0.38)
    parser.add_argument("--cosupport-dominance-ratio-min", type=float, default=1.0)
    parser.add_argument("--disable-mask-cosupport", action="store_true")
    parser.add_argument("--enable-cosupport-native-gate", action="store_true")
    parser.add_argument("--cosupport-native-min-support", type=int, default=20)
    parser.add_argument("--native-carrier-rows", default=None)
    parser.add_argument("--disable-native-frame-mask-projection", action="store_true")
    parser.add_argument("--native-boundary-min-support", type=int, default=100)
    parser.add_argument("--native-boundary-min-candidate-ratio", type=float, default=0.10)
    parser.add_argument("--native-boundary-min-jaccard", type=float, default=0.01)
    parser.add_argument("--native-boundary-min-shared-frame-masks", type=int, default=3)
    parser.add_argument("--enable-native-uv-projection", action="store_true")
    parser.add_argument("--native-uv-min-support", type=int, default=20)
    parser.add_argument("--native-uv-min-candidate-ratio", type=float, default=0.10)
    parser.add_argument("--native-uv-min-jaccard", type=float, default=0.0)
    parser.add_argument("--native-uv-min-mean-iou", type=float, default=0.05)
    parser.add_argument("--native-uv-max-center-dist", type=float, default=0.10)
    parser.add_argument("--native-uv-min-shared-frames", type=int, default=3)
    parser.add_argument("--enable-native-history-mask-projection", action="store_true")
    parser.add_argument("--native-history-mask-min-support", type=int, default=100)
    parser.add_argument("--native-history-mask-min-ratio", type=float, default=0.90)
    parser.add_argument("--native-history-mask-min-dominance", type=float, default=1.5)
    parser.add_argument("--native-history-mask-min-mask-ratio", type=float, default=0.0)
    parser.add_argument("--enable-native-history-mask-component-gate", action="store_true")
    parser.add_argument("--native-history-mask-component-min-support", type=int, default=1)
    parser.add_argument("--enable-native-history-mask-component-accumulation-gate", action="store_true")
    parser.add_argument("--native-history-mask-component-accumulation-min-support", type=int, default=1)
    parser.add_argument("--native-history-mask-component-accumulation-min-masks", type=int, default=2)
    parser.add_argument("--native-history-mask-component-accumulation-min-frames", type=int, default=2)
    parser.add_argument("--enable-native-history-mask-component-support-gate", action="store_true")
    parser.add_argument("--native-history-mask-component-max-selected-rank", type=int, default=0)
    parser.add_argument("--native-history-mask-component-min-w-visible", type=float, default=0.0)
    parser.add_argument("--native-history-mask-component-min-r-mask", type=float, default=0.0)
    parser.add_argument("--native-history-mask-component-require-dominant", action="store_true")
    parser.add_argument("--enable-native-history-mask-cannot-link-guard", action="store_true")
    parser.add_argument("--native-history-mask-other-seed-min-support", type=int, default=1)
    parser.add_argument("--native-history-mask-other-seed-min-ratio", type=float, default=0.05)
    parser.add_argument("--native-history-mask-second-native-min-support", type=int, default=1)
    parser.add_argument("--native-history-mask-second-native-min-ratio", type=float, default=0.05)
    parser.add_argument("--enable-native-history-mask-semantic-guard", action="store_true")
    parser.add_argument("--native-history-mask-semantic-backend", default="colorhist", choices=["colorhist", "dinov2_timm"])
    parser.add_argument("--native-history-mask-semantic-min-cosine", type=float, default=0.94)
    parser.add_argument("--native-history-mask-semantic-device", default="cpu")
    parser.add_argument("--native-history-mask-semantic-checkpoint", default=None)
    parser.add_argument("--native-history-mask-semantic-short-side", type=int, default=518)
    args = parser.parse_args()
    payload = build_v55_history_update(
        history_evidence_roles=tuple(role.strip() for role in args.history_evidence_roles.split(",") if role.strip()),
        cosupport_seed_ratio_min=args.cosupport_seed_ratio_min,
        cosupport_dominance_ratio_min=args.cosupport_dominance_ratio_min,
        enable_mask_cosupport=not args.disable_mask_cosupport,
        enable_cosupport_native_gate=args.enable_cosupport_native_gate,
        cosupport_native_min_support=args.cosupport_native_min_support,
        native_carrier_rows_path=args.native_carrier_rows,
        enable_native_frame_mask_projection=not args.disable_native_frame_mask_projection,
        native_boundary_min_support=args.native_boundary_min_support,
        native_boundary_min_candidate_ratio=args.native_boundary_min_candidate_ratio,
        native_boundary_min_jaccard=args.native_boundary_min_jaccard,
        native_boundary_min_shared_frame_masks=args.native_boundary_min_shared_frame_masks,
        enable_native_uv_projection=args.enable_native_uv_projection,
        native_uv_min_support=args.native_uv_min_support,
        native_uv_min_candidate_ratio=args.native_uv_min_candidate_ratio,
        native_uv_min_jaccard=args.native_uv_min_jaccard,
        native_uv_min_mean_iou=args.native_uv_min_mean_iou,
        native_uv_max_center_dist=args.native_uv_max_center_dist,
        native_uv_min_shared_frames=args.native_uv_min_shared_frames,
        enable_native_history_mask_projection=args.enable_native_history_mask_projection,
        native_history_mask_min_support=args.native_history_mask_min_support,
        native_history_mask_min_ratio=args.native_history_mask_min_ratio,
        native_history_mask_min_dominance=args.native_history_mask_min_dominance,
        native_history_mask_min_mask_ratio=args.native_history_mask_min_mask_ratio,
        enable_native_history_mask_component_gate=args.enable_native_history_mask_component_gate,
        native_history_mask_component_min_support=args.native_history_mask_component_min_support,
        enable_native_history_mask_component_accumulation_gate=args.enable_native_history_mask_component_accumulation_gate,
        native_history_mask_component_accumulation_min_support=args.native_history_mask_component_accumulation_min_support,
        native_history_mask_component_accumulation_min_masks=args.native_history_mask_component_accumulation_min_masks,
        native_history_mask_component_accumulation_min_frames=args.native_history_mask_component_accumulation_min_frames,
        enable_native_history_mask_component_support_gate=args.enable_native_history_mask_component_support_gate,
        native_history_mask_component_max_selected_rank=args.native_history_mask_component_max_selected_rank,
        native_history_mask_component_min_w_visible=args.native_history_mask_component_min_w_visible,
        native_history_mask_component_min_r_mask=args.native_history_mask_component_min_r_mask,
        native_history_mask_component_require_dominant=args.native_history_mask_component_require_dominant,
        enable_native_history_mask_cannot_link_guard=args.enable_native_history_mask_cannot_link_guard,
        native_history_mask_other_seed_min_support=args.native_history_mask_other_seed_min_support,
        native_history_mask_other_seed_min_ratio=args.native_history_mask_other_seed_min_ratio,
        native_history_mask_second_native_min_support=args.native_history_mask_second_native_min_support,
        native_history_mask_second_native_min_ratio=args.native_history_mask_second_native_min_ratio,
        enable_native_history_mask_semantic_guard=args.enable_native_history_mask_semantic_guard,
        native_history_mask_semantic_backend=args.native_history_mask_semantic_backend,
        native_history_mask_semantic_min_cosine=args.native_history_mask_semantic_min_cosine,
        native_history_mask_semantic_device=args.native_history_mask_semantic_device,
        native_history_mask_semantic_checkpoint=args.native_history_mask_semantic_checkpoint,
        native_history_mask_semantic_short_side=args.native_history_mask_semantic_short_side,
    )
    write_v55_history_update(args.output_root, payload, visualization_root=args.visualization_root)
    summary = payload["summary"]
    print(
        {
            "history_update": f"{args.output_root}/history_update_summary.json",
            "gate": summary["gate"],
            "history_object_count": summary["history_object_count"],
            "confirmed_update_count": summary["confirmed_update_count"],
            "partial_update_count": summary["partial_update_count"],
            "history_temporal_span_mean": summary["history_temporal_span_mean"],
            "real_minus_shuffled_ARI": summary["real_minus_shuffled_ARI"],
            "real_minus_no_temporal_ARI": summary["real_minus_no_temporal_ARI"],
        }
    )


if __name__ == "__main__":
    main()
