from __future__ import annotations

import argparse

from stream4d_native.v56_core_update_projection_component_guard import (
    build_v56_projection_component_guard_core_update,
    write_v56_projection_component_guard_core_update,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v56 C4c projection component guard from C3 updates.")
    parser.add_argument("--output-root", default="outputs/audit/v56_core_update_c4c_projection_component_guard")
    parser.add_argument("--boundary-component-min-support", type=int, default=1)
    parser.add_argument("--boundary-component-min-ratio", type=float, default=0.0)
    parser.add_argument("--uv-component-min-support", type=int, default=1)
    parser.add_argument("--uv-component-min-ratio", type=float, default=0.0)
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument("--uv-only", action="store_true")
    args = parser.parse_args()
    include_boundary = not args.uv_only
    include_uv = not args.boundary_only
    payload = build_v56_projection_component_guard_core_update(
        boundary_component_min_support=args.boundary_component_min_support,
        boundary_component_min_ratio=args.boundary_component_min_ratio,
        uv_component_min_support=args.uv_component_min_support,
        uv_component_min_ratio=args.uv_component_min_ratio,
        include_boundary=include_boundary,
        include_uv=include_uv,
    )
    write_v56_projection_component_guard_core_update(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/core_update_summary.json",
            "gate": summary["gate"],
            "source_c3_confirmed_update_count": summary["source_c3_confirmed_update_count"],
            "confirmed_update_count": summary["confirmed_update_count"],
            "confirmed_added_component_count": summary["confirmed_added_component_count"],
            "filtered_component_count": summary["filtered_component_count"],
            "component_guard_reject_count": summary["component_guard_reject_count"],
            "history_temporal_span_mean": summary["history_temporal_span_mean"],
            "history_ARI": summary["history_ARI"],
            "history_purity": summary["history_purity"],
            "history_completeness": summary["history_completeness"],
            "update_precision_diagnostic": summary["update_precision_diagnostic"],
            "real_minus_shuffled_ARI": summary["real_minus_shuffled_ARI"],
            "real_minus_no_temporal_ARI": summary["real_minus_no_temporal_ARI"],
        }
    )


if __name__ == "__main__":
    main()
