from __future__ import annotations

import argparse

from stream4d_native.v64r2_main_fact_lock import build_v64r2_main_fact_lock, write_v64r2_main_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase A0 main ownership fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_phaseA0_main_fact_lock")
    args = parser.parse_args()
    payload = build_v64r2_main_fact_lock()
    write_v64r2_main_fact_lock(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/main_fact_lock_summary.json",
            "main_ownership_status": payload["summary"]["main_ownership_status"],
            "gate": payload["gate"],
        }
    )


if __name__ == "__main__":
    main()
