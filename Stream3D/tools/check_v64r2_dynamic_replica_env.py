from __future__ import annotations

import argparse

from stream4d_native.v64r2_dynamic_env import build_v64r2_dynamic_env, write_v64r2_dynamic_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase C0 Dynamic Replica environment check.")
    parser.add_argument("--data-root", default="data/dynamic-replica/v2")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_dynamic_env")
    parser.add_argument("--max-annotation-rows", type=int, default=0)
    args = parser.parse_args()
    payload = build_v64r2_dynamic_env(
        data_root=args.data_root,
        split=args.split,
        max_annotation_rows=args.max_annotation_rows,
    )
    write_v64r2_dynamic_env(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/dynamic_env_summary.json",
            "dyn_level": summary["dyn_level_label"],
            "annotation_items_scanned": summary["annotation_items_scanned"],
            "rgb_frames_exist": summary["rgb_frames_exist"],
            "depth_frames_exist": summary["depth_frames_exist"],
            "instance_masks_exist": summary["instance_masks_exist"],
            "object_ids_exist": summary["object_ids_exist"],
            "trajectories_exist": summary["trajectories_exist"],
            "can_report_official_object_tracking": summary["can_report_official_object_tracking"],
        }
    )


if __name__ == "__main__":
    main()
