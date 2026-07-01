from __future__ import annotations

import argparse

from stream4d_native.v56_quarantine_promotion_veto import (
    build_v56_quarantine_promotion_veto,
    write_v56_quarantine_promotion_veto,
)


def _parse_ratio_thresholds(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v56 quarantine veto diagnostics over promotion rows.")
    parser.add_argument("--promotion-rows", required=True)
    parser.add_argument("--quarantine-rows", default="outputs/audit/v56_quarantine/quarantine_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v56_quarantine_promotion_veto")
    parser.add_argument("--ratio-thresholds", default="0.0,0.02,0.04,0.08,0.12")
    parser.add_argument("--any-shared-history", action="store_true")
    args = parser.parse_args()
    payload = build_v56_quarantine_promotion_veto(
        promotion_rows_path=args.promotion_rows,
        quarantine_rows_path=args.quarantine_rows,
        ratio_thresholds=_parse_ratio_thresholds(args.ratio_thresholds),
        same_anchor_only=not args.any_shared_history,
    )
    write_v56_quarantine_promotion_veto(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/quarantine_promotion_veto_summary.json",
            "gate": summary["gate"],
            "promoted_component_count": summary["promoted_component_count"],
            "false_promotion_count": summary["false_promotion_count"],
            "best_safe_variant": summary["best_safe_variant"],
            "best_reduction_variant": summary["best_reduction_variant"],
            "best_reduction_false_promotion_reduction": summary["best_reduction_false_promotion_reduction"],
            "best_reduction_expanded_completeness_drop_proxy": summary[
                "best_reduction_expanded_completeness_drop_proxy"
            ],
        }
    )


if __name__ == "__main__":
    main()

