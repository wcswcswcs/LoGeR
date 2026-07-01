from __future__ import annotations

import argparse

from stream4d_native.v64r2_dynamic_adapter import build_v64r2_dynamic_adapter, write_v64r2_dynamic_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase C1 Dynamic Replica method adapter.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_dynamic_adapter")
    args = parser.parse_args()
    payload = build_v64r2_dynamic_adapter()
    write_v64r2_dynamic_adapter(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/adapter_summary.json",
            "method_adapter_status": summary["method_adapter_status"],
            "gate": summary["gate"],
            "blocked_reason": summary["blocked_reason"],
        }
    )


if __name__ == "__main__":
    main()
