from __future__ import annotations

import argparse

from stream4d_native.v50_stage1 import build_v50_fact_lock, write_v50_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v50 Phase 0 fact lock and AP contract.")
    parser.add_argument("--output-root", default="outputs/audit/v50_fact_lock")
    args = parser.parse_args()
    payload = build_v50_fact_lock()
    write_v50_fact_lock(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/fact_lock.json",
            "gate": payload["gate"],
            "missing_required": payload["missing_required"],
            "v49_AP_status": payload["fact_map"].get("v49_AP_status"),
        }
    )


if __name__ == "__main__":
    main()
