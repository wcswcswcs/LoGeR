#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_visualization_export import CASE_ROOT, VIS_ROOT, build_v65_visual_casebook


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v65 AP/geometry/visual casebook.")
    parser.add_argument("--case-root", default=CASE_ROOT)
    parser.add_argument("--vis-root", default=VIS_ROOT)
    args = parser.parse_args()
    payload = build_v65_visual_casebook(args.case_root, args.vis_root)
    print(
        {
            "summary": f"{args.case_root}/casebook_summary.json",
            "gate": payload["summary"]["gate"],
            "case_count": payload["summary"]["case_count"],
            "uses_fallback_screenshots": payload["summary"]["uses_fallback_screenshots"],
        }
    )


if __name__ == "__main__":
    main()
