from __future__ import annotations

import argparse
from pathlib import Path

from stream4d_native.v47_carrier_observation_table import build_observation_tables
from stream4d_native.v47_common import ROOT, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v47 carrier/mask observation tables.")
    parser.add_argument("--carrier-cache-root", default="outputs/stream4d_debug_v47_stride1_d5_probe5_mf32")
    parser.add_argument("--scenes", default="scene0011_00,scene0030_00,scene0050_00,scene0081_01,scene0591_00")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--feature-backend", default="colorhist_fallback")
    parser.add_argument("--output-root", default="outputs/audit/v47_observation_tables")
    args = parser.parse_args()
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    payload = build_observation_tables(
        carrier_cache_root=ROOT / str(args.carrier_cache_root),
        scenes=scenes,
        visibility_threshold=float(args.visibility_threshold),
        confidence_threshold=float(args.confidence_threshold),
        min_mask_area=int(args.min_mask_area),
        feature_backend=str(args.feature_backend),
    )
    out = ROOT / str(args.output_root)
    write_csv(out / "carrier_observation_table.csv", payload["carrier_rows"])
    write_csv(out / "mask_observation_table.csv", payload["mask_rows"])
    write_csv(out / "observation_window_rows.csv", payload["window_rows"])
    write_json(out / "observation_table_summary.json", payload["summary"])
    print({"summary": str(out / "observation_table_summary.json"), "gate": payload["summary"]["gate"]})


if __name__ == "__main__":
    main()

