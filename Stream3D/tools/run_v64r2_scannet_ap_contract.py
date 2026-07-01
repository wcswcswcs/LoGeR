from __future__ import annotations

import argparse

from stream4d_native.v64r2_scannet_exporters import build_v64r2_ap_contract, write_v64r2_ap_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase B0 AP exporter policy contract.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_ap_contract")
    args = parser.parse_args()
    payload = build_v64r2_ap_contract()
    write_v64r2_ap_contract(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/ap_export_contract.json",
            "ap_status_before_probe": summary["ap_status_before_probe"],
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
