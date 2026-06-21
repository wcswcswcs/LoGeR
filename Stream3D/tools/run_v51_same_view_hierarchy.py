from __future__ import annotations

import argparse

from stream4d_native.v51_same_view_hierarchy import build_v51_same_view_hierarchy, write_v51_same_view_hierarchy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v51-r2 same-view hierarchy from real overlap mask stacks.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_same_view_hierarchy")
    parser.add_argument("--contain-threshold", type=float, default=0.85)
    parser.add_argument("--area-ratio-threshold", type=float, default=1.30)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.80)
    parser.add_argument("--max-relation-rows", type=int, default=50000)
    args = parser.parse_args()
    payload = build_v51_same_view_hierarchy(
        input_root=args.input_root,
        contain_threshold=args.contain_threshold,
        area_ratio_threshold=args.area_ratio_threshold,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        max_relation_rows=args.max_relation_rows,
    )
    write_v51_same_view_hierarchy(args.output_root, payload)
    print({"summary": f"{args.output_root}/hierarchy_summary.json", "gate": payload["gate"], "metrics": payload["summary"]})


if __name__ == "__main__":
    main()
