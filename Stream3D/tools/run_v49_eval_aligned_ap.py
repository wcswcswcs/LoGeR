from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_eval_aligned_ap, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 11 eval-aligned AP diagnostic gate.")
    parser.add_argument("--output-root", default="outputs/audit/v49_eval_aligned_ap")
    args = parser.parse_args()
    payload = build_eval_aligned_ap()
    write_bundle(args.output_root, "eval_aligned_ap_summary", payload, {"eval_aligned_ap_rows": payload["ap_rows"]})
    print({"summary": f"{args.output_root}/eval_aligned_ap_summary.json", "gate": payload["gate"]})


if __name__ == "__main__":
    main()
