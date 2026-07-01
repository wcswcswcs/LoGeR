from __future__ import annotations

import argparse

from stream4d_native.v47_common import ROOT, write_csv, write_json
from stream4d_native.v64r2_dynamic_metrics import build_v64r2_dynamic_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v64-r2 Phase C2 Dynamic Replica metrics by data level.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_dynamic_metrics")
    args = parser.parse_args()
    payload = build_v64r2_dynamic_metrics(split_name="available_level")
    out = ROOT / args.output_root
    write_json(out / "dynamic_metric_summary.json", payload["summary"])
    write_csv(out / "dynamic_metric_rows.csv", payload["dynamic_metric_rows"])
    print(
        {
            "summary": f"{args.output_root}/dynamic_metric_summary.json",
            "dynamic_metric_status": payload["summary"]["dynamic_metric_status"],
            "dyn_level": payload["summary"]["dyn_level_label"],
            "blocked_reason": payload["summary"]["blocked_reason"],
        }
    )


if __name__ == "__main__":
    main()
