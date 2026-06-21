from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_full_stage1, write_v50_full_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 9 full Stage-1 controls/significance summary.")
    parser.add_argument("--output-root", default="outputs/audit/v50_full_stage1")
    args = parser.parse_args()
    payload = build_v50_full_stage1()
    write_v50_full_stage1(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/full_stage1_summary.json",
            "gate": payload["gate"],
            "failure_label": payload["failure_label"],
            "partial_label": payload["partial_label"],
        }
    )


if __name__ == "__main__":
    main()
