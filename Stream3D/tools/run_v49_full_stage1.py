from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_full_stage1, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 9 full Stage-1 final gate.")
    parser.add_argument("--output-root", default="outputs/audit/v49_full_stage1")
    args = parser.parse_args()
    payload = build_full_stage1()
    write_bundle(
        args.output_root,
        "full_stage1_summary",
        payload,
        {"v49_stage1_variant_rows": payload["stage1_rows"], "v49_stage1_gate_rows": payload["stage1_gate_rows"]},
    )
    print({"summary": f"{args.output_root}/full_stage1_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
