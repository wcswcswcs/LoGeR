from __future__ import annotations

import argparse

from stream4d_native.v64r2_dynamic_metrics import build_v64r2_dynamic_metrics, write_v64r2_dynamic_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase C3 Dynamic Replica full split metrics.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_dynamic_full")
    args = parser.parse_args()
    payload = build_v64r2_dynamic_metrics(split_name="full")
    write_v64r2_dynamic_metrics(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/dynamic_full_summary.json",
            "dynamic_metric_status": summary["dynamic_metric_status"],
            "dyn_level": summary["dyn_level_label"],
            "gate": summary["gate"],
            "blocked_reason": summary["blocked_reason"],
        }
    )


if __name__ == "__main__":
    main()
