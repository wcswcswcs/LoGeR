from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_stream3d_parity import (
    build_v65_stream3d_parity,
    run_v65_stream3d_parity,
    write_v65_stream3d_parity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v65 Stream3D parity rows.")
    parser.add_argument("--output-root", default="outputs/audit/v65_stream3d_parity")
    parser.add_argument("--skip-recompute", action="store_true")
    args = parser.parse_args()
    command_rows = [] if args.skip_recompute else run_v65_stream3d_parity(audit_root=args.output_root)
    payload = build_v65_stream3d_parity(command_rows=command_rows)
    write_v65_stream3d_parity(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/stream3d_parity_summary.json",
            "gate": payload["summary"]["gate"],
            "failed_command_count": payload["summary"]["failed_command_count"],
            "row_count": payload["summary"]["row_count"],
        }
    )


if __name__ == "__main__":
    main()
