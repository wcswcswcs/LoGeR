from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_ap_failure_decomp import build_v65_ap_failure_decomp, write_v65_ap_failure_decomp


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v65 AP failure decomposition from current prediction/support files.")
    parser.add_argument("--output-root", default="outputs/audit/v65_ap_failure_decomp")
    args = parser.parse_args()
    payload = build_v65_ap_failure_decomp()
    write_v65_ap_failure_decomp(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/failure_summary.json",
            "gate": payload["summary"]["gate"],
            "failure_row_count": payload["summary"]["failure_row_count"],
            "top_failure_category": payload["summary"]["top_failure_category"],
        }
    )


if __name__ == "__main__":
    main()
