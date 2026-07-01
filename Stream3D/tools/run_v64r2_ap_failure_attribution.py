from __future__ import annotations

import argparse

from stream4d_native.v64r2_ap_failure_attribution import (
    build_v64r2_ap_failure_attribution,
    write_v64r2_ap_failure_attribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase B2 AP failure attribution.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_ap_failure_attribution")
    args = parser.parse_args()
    payload = build_v64r2_ap_failure_attribution()
    write_v64r2_ap_failure_attribution(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/failure_summary.json",
            "top_failure_category": summary["top_failure_category"],
            "attribution_coverage": summary["attribution_coverage"],
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
