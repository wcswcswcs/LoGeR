from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_ap_contract import build_v65_ap_contract, run_v65_ap_recompute, write_v65_ap_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute v65 AP rows and write AP contract outputs.")
    parser.add_argument("--output-root", default="outputs/audit/v65_ap_contract")
    parser.add_argument("--skip-recompute", action="store_true")
    args = parser.parse_args()
    command_rows = [] if args.skip_recompute else run_v65_ap_recompute(audit_root=args.output_root)
    payload = build_v65_ap_contract(command_rows=command_rows)
    write_v65_ap_contract(args.output_root, payload)
    failed = [row for row in command_rows if int(row.get("returncode", 0)) != 0]
    print(
        {
            "summary": f"{args.output_root}/ap_contract_summary.json",
            "row_count": payload["summary"]["row_count"],
            "gate": payload["summary"]["gate"],
            "failed_commands": len(failed),
            "method_safe_rows_with_AP": payload["summary"]["method_safe_rows_with_AP"],
        }
    )


if __name__ == "__main__":
    main()
