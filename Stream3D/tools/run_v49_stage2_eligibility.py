from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_stage2_eligibility, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 12 Stage-2 eligibility gate.")
    parser.add_argument("--output-root", default="outputs/audit/v49_stage2")
    args = parser.parse_args()
    payload = build_stage2_eligibility()
    write_bundle(args.output_root, "stage2_eligibility_summary", payload, {"stage2_rows": payload["stage2_rows"]})
    print({"summary": f"{args.output_root}/stage2_eligibility_summary.json", "entry_gate": payload["entry_gate"]})


if __name__ == "__main__":
    main()
