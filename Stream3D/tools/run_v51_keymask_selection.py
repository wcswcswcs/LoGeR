from __future__ import annotations

import argparse

from stream4d_native.v51_keymask_selection import build_v51_keymask_selection, write_v51_keymask_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Select v51-r2 representative key masks from SAM2 raw proposal support.")
    parser.add_argument("--raw-mask-root", default="outputs/remask/v51_r2/sam2_tiny_probe5_4f_p64_crop1_relaxed")
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_key_masks")
    parser.add_argument("--carrier-rows-path", default="outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv")
    parser.add_argument("--vote-rows-path", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv")
    parser.add_argument("--scenes", default="scene0011_00,scene0030_00,scene0050_00,scene0081_01,scene0591_00")
    parser.add_argument("--frame-ids", default="0,10,20,30")
    parser.add_argument("--component-min-carriers", type=int, default=1)
    parser.add_argument("--underseg-component-threshold", type=int, default=5)
    args = parser.parse_args()
    payload = build_v51_keymask_selection(
        raw_mask_root=args.raw_mask_root,
        carrier_rows_path=args.carrier_rows_path,
        vote_rows_path=args.vote_rows_path,
        scenes=args.scenes,
        frame_ids=args.frame_ids,
        component_min_carriers=args.component_min_carriers,
        underseg_component_threshold=args.underseg_component_threshold,
    )
    write_v51_keymask_selection(args.output_root, payload)
    print({"summary": f"{args.output_root}/keymask_summary.json", "gate": payload["gate"], "metrics": payload["summary"]})


if __name__ == "__main__":
    main()
