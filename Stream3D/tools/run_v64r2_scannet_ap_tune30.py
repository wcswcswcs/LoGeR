from __future__ import annotations

import argparse

from stream4d_native.v64r2_scannet_ap_eval import (
    build_v64r2_scannet_ap_locked_split,
    write_v64r2_scannet_ap_locked_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase B3 ScanNet AP tune30 decision.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_scannet_ap_tune30")
    args = parser.parse_args()
    payload = build_v64r2_scannet_ap_locked_split(split="tune30")
    write_v64r2_scannet_ap_locked_split(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/ap_tune_summary.json",
            "status": payload["summary"]["status"],
            "blocked_reason": payload["summary"]["blocked_reason"],
        }
    )


if __name__ == "__main__":
    main()
