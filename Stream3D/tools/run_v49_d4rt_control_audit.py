from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_d4rt_control_audit, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 Phase 8 D4RT control audit.")
    parser.add_argument("--output-root", default="outputs/audit/v49_d4rt_controls")
    args = parser.parse_args()
    payload = build_d4rt_control_audit()
    write_bundle(args.output_root, "d4rt_control_audit_summary", payload, {"d4rt_control_rows": payload["control_rows"]})
    print({"summary": f"{args.output_root}/d4rt_control_audit_summary.json", "gate": payload["gate"], "failure_label": payload["failure_label"]})


if __name__ == "__main__":
    main()
