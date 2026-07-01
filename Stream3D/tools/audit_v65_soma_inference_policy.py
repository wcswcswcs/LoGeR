#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_soma_inference_audit import (
    DEFAULT_OUTPUT_ROOT,
    build_v65_soma_inference_audit,
    write_v65_soma_inference_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SOMA artifacts for no-GT method inference policy.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    payload = build_v65_soma_inference_audit()
    paths = write_v65_soma_inference_audit(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": paths["summary"],
            "scanned_rows": paths["scanned_rows"],
            "violation_rows": paths["violation_rows"],
            "record_count": summary["record_count"],
            "policy_violation_count": summary["policy_violation_count"],
            "method_inference_gt_geometry_record_count": summary["method_inference_gt_geometry_record_count"],
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
