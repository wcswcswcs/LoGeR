from __future__ import annotations

import argparse

from stream4d_native.v64r2_native_contract import build_v64r2_native_contract, write_v64r2_native_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase A1 native object/material contract export.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_native_contract")
    args = parser.parse_args()
    payload = build_v64r2_native_contract()
    write_v64r2_native_contract(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/native_contract_summary.json",
            "object_count": summary["object_count"],
            "material_count": summary["material_count"],
            "carrier_level_available": summary["carrier_level_available"],
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
