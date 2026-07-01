#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_a9_method_safe_materializer import (  # noqa: E402
    A9_ROOT,
    build_v65_a9_method_safe_materializer,
    write_v65_a9_method_safe_materializer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attempt v65 A9 method-safe native AP materialization.")
    parser.add_argument("--output-root", default=A9_ROOT)
    parser.add_argument("--no-rerun-native-carrier", action="store_true")
    args = parser.parse_args()
    payload = build_v65_a9_method_safe_materializer(
        output_root=args.output_root,
        rerun_native_carrier=not args.no_rerun_native_carrier,
    )
    write_v65_a9_method_safe_materializer(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/a9_materializer_summary.json",
            "status": summary["status"],
            "method_safe_native_support_available": summary["method_safe_native_support_available"],
            "scan_ap_join_key_available": summary["scan_ap_join_key_available"],
            "method_safe_ap_available": summary["method_safe_ap_available"],
        }
    )


if __name__ == "__main__":
    main()
