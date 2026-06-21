from __future__ import annotations

import argparse

from stream4d_native.v51_mask_bank_filter import filter_mask_bank_by_containment


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a v51 mask bank by non-GT containment geometry.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--contain-threshold", type=float, default=0.85)
    parser.add_argument("--area-ratio-threshold", type=float, default=1.30)
    parser.add_argument("--min-masks-per-frame", type=int, default=10)
    args = parser.parse_args()
    payload = filter_mask_bank_by_containment(
        input_root=args.input_root,
        output_root=args.output_root,
        contain_threshold=args.contain_threshold,
        area_ratio_threshold=args.area_ratio_threshold,
        min_masks_per_frame=args.min_masks_per_frame,
    )
    print(
        {
            "summary": f"{args.output_root}/mask_bank_filter_summary.json",
            "input_mask_count": payload["input_mask_count"],
            "output_mask_count": payload["output_mask_count"],
            "mean_output_masks_per_frame": payload["mean_output_masks_per_frame"],
            "containment_pair_count_before_filter": payload["containment_pair_count_before_filter"],
        }
    )


if __name__ == "__main__":
    main()
