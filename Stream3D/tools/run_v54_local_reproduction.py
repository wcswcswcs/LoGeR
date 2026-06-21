from __future__ import annotations

import argparse

from stream4d_native.v54_local_reproduction import build_v54_local_reproduction, write_v54_local_reproduction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v54 chunk-local reproduction summary.")
    parser.add_argument("--support-rows", required=True)
    parser.add_argument("--chunk-component-rows", required=True)
    parser.add_argument("--chunk-mask-rows", required=True)
    parser.add_argument("--objectlet-summary", required=True)
    parser.add_argument("--objectlet-rows", required=True)
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--native-summary", default=None)
    parser.add_argument(
        "--enable-weak-support-restitution",
        action="store_true",
        help="Add diagnostic W1 rows where mask-only representative support fills only method-uncovered components.",
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    payload = build_v54_local_reproduction(
        support_rows_path=args.support_rows,
        chunk_component_rows_path=args.chunk_component_rows,
        chunk_mask_rows_path=args.chunk_mask_rows,
        objectlet_summary_path=args.objectlet_summary,
        objectlet_rows_path=args.objectlet_rows,
        support_variant=args.support_variant,
        native_summary_path=args.native_summary,
        enable_weak_support_restitution=args.enable_weak_support_restitution,
    )
    write_v54_local_reproduction(args.output_root, payload)
    summary = payload["summary"]
    best = summary.get("best_method_row") or {}
    weak = summary.get("mask_only_restitution_row") or {}
    weak_support = summary.get("best_weak_support_row") or {}
    print(
        {
            "summary": f"{args.output_root}/local_reproduction_summary.json",
            "best_method_variant": summary.get("best_method_variant"),
            "local_gate": summary.get("local_gate"),
            "best_local_ARI_mean": best.get("local_ARI_mean"),
            "best_local_purity_mean": best.get("local_purity_mean"),
            "best_local_completeness_mean": best.get("local_completeness_mean"),
            "best_local_conflict_rate_mean": best.get("local_conflict_rate_mean"),
            "weak_mask_only_restitution_gate": summary.get("weak_mask_only_restitution_gate"),
            "mask_only_local_ARI_mean": weak.get("local_ARI_mean"),
            "mask_only_local_purity_mean": weak.get("local_purity_mean"),
            "mask_only_local_completeness_mean": weak.get("local_completeness_mean"),
            "weak_support_restitution_gate": summary.get("weak_support_restitution_gate"),
            "best_weak_support_variant": summary.get("best_weak_support_variant"),
            "weak_support_local_ARI_mean": weak_support.get("local_ARI_mean"),
            "weak_support_local_purity_mean": weak_support.get("local_purity_mean"),
            "weak_support_local_completeness_mean": weak_support.get("local_completeness_mean"),
        }
    )


if __name__ == "__main__":
    main()
