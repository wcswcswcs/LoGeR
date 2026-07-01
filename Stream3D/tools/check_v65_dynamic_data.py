#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_dynamic_data import DYNAMIC_ROOT, build_v65_dynamic_data, write_v65_dynamic_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v65 Dynamic Replica data-level and metric-permission check.")
    parser.add_argument("--data-root", default="data/dynamic-replica/v2")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output-root", default=DYNAMIC_ROOT)
    parser.add_argument("--max-annotation-rows", type=int, default=0)
    args = parser.parse_args()
    payload = build_v65_dynamic_data(
        data_root=args.data_root,
        split=args.split,
        max_annotation_rows=args.max_annotation_rows,
    )
    write_v65_dynamic_data(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/dynamic_data_summary.json",
            "dyn_level": summary["dyn_level_label"],
            "dynamic_status": summary["dynamic_status"],
            "annotation_items_scanned": summary["annotation_items_scanned"],
            "actual_images_count": summary["actual_images_count"],
            "actual_depth_count": summary["actual_depth_count"],
            "actual_mask_count": summary["actual_mask_count"],
            "actual_instance_id_map_count": summary["actual_instance_id_map_count"],
            "actual_trajectory_count": summary["actual_trajectory_count"],
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
