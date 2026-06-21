from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_stage2_eligibility, write_v50_stage2_eligibility


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 11 Stage-2 eligibility policy.")
    parser.add_argument("--output-root", default="outputs/audit/v50_stage2")
    args = parser.parse_args()
    payload = build_v50_stage2_eligibility()
    write_v50_stage2_eligibility(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/stage2_eligibility_summary.json",
            "entry_gate": payload["entry_gate"],
            "stage2_allowed": payload["stage2_allowed"],
            "stage2_block_reason": payload["stage2_block_reason"],
        }
    )


if __name__ == "__main__":
    main()
