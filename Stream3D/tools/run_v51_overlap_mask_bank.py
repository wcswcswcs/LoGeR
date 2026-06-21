from __future__ import annotations

import argparse

from stream4d_native.v51_overlap_mask_bank import build_v51_overlap_mask_bank, write_v51_overlap_mask_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stream4D v51-r2 OverlapMaskBank gate for an NPZ mask-stack root.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_overlap_mask_bank")
    parser.add_argument("--max-files", type=int, default=32)
    args = parser.parse_args()
    payload = build_v51_overlap_mask_bank(args.input_root, max_files=args.max_files)
    write_v51_overlap_mask_bank(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/overlap_mask_bank_summary.json",
            "gate": payload["gate"],
            "metrics": payload["summary"],
        }
    )


if __name__ == "__main__":
    main()
