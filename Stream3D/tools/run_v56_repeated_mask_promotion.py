from __future__ import annotations

import argparse

from stream4d_native.v56_promotion_repeated_masks import (
    build_v56_repeated_mask_promotion,
    write_v56_repeated_mask_promotion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v56 repeated-independent-mask promotion diagnostic.")
    parser.add_argument("--output-root", default="outputs/audit/v56_promotion_repeated_masks")
    parser.add_argument("--min-independent-chunks", type=int, default=2)
    parser.add_argument("--min-co-support-masks", type=int, default=2)
    parser.add_argument("--min-component-support-count", type=int, default=1)
    parser.add_argument("--require-no-competing-history", action="store_true")
    parser.add_argument("--max-competing-history-count", type=int, default=0)
    parser.add_argument("--exclude-multi-history-tentative-components", action="store_true")
    args = parser.parse_args()
    payload = build_v56_repeated_mask_promotion(
        min_independent_chunks=args.min_independent_chunks,
        min_co_support_masks=args.min_co_support_masks,
        min_component_support_count=args.min_component_support_count,
        require_no_competing_history=args.require_no_competing_history,
        max_competing_history_count=args.max_competing_history_count,
        exclude_multi_history_tentative_components=args.exclude_multi_history_tentative_components,
    )
    write_v56_repeated_mask_promotion(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/promotion_summary.json",
            "gate": summary["gate"],
            "promotion_candidate_count": summary["promotion_candidate_count"],
            "eligible_promotion_candidate_count": summary["eligible_promotion_candidate_count"],
            "quarantine_candidate_reject_count": summary["quarantine_candidate_reject_count"],
            "promoted_component_count": summary["promoted_component_count"],
            "promotion_precision_diagnostic": summary["promotion_precision_diagnostic"],
            "false_promotion_count": summary["false_promotion_count"],
            "confirmed_core_ARI": summary["confirmed_core_ARI"],
            "confirmed_core_purity": summary["confirmed_core_purity"],
            "confirmed_core_completeness": summary["confirmed_core_completeness"],
            "confirmed_core_completeness_gain_vs_P0": summary["confirmed_core_completeness_gain_vs_P0"],
            "real_minus_shuffled_ARI": summary["real_minus_shuffled_ARI"],
            "real_minus_no_temporal_ARI": summary["real_minus_no_temporal_ARI"],
            "real_minus_shuffled_ARI_gain_vs_P0": summary["real_minus_shuffled_ARI_gain_vs_P0"],
            "real_minus_no_temporal_ARI_gain_vs_P0": summary["real_minus_no_temporal_ARI_gain_vs_P0"],
        }
    )


if __name__ == "__main__":
    main()
