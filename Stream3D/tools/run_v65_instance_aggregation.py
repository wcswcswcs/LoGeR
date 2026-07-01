from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_instance_aggregation import (
    build_v65_instance_aggregation,
    run_v65_instance_aggregation,
    write_v65_instance_aggregation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v65 D4RT fragment-to-instance aggregation diagnostics.")
    parser.add_argument("--output-root", default="outputs/audit/v65_instance_aggregation")
    parser.add_argument("--skip-recompute", action="store_true")
    args = parser.parse_args()
    command_rows = [] if args.skip_recompute else run_v65_instance_aggregation()
    payload = build_v65_instance_aggregation(command_rows=command_rows)
    write_v65_instance_aggregation(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/aggregation_summary.json",
            "gate": payload["summary"]["gate"],
            "blocker": payload["summary"]["blocker"],
            "failed_command_count": payload["summary"]["failed_command_count"],
        }
    )


if __name__ == "__main__":
    main()
