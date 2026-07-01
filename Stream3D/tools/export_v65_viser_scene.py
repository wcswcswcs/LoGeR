#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_visualization_export import VIS_ROOT, export_v65_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v65 visualization scene data, bookmarks, and screenshots.")
    parser.add_argument("--output-root", default=VIS_ROOT)
    args = parser.parse_args()
    payload = export_v65_visualization(args.output_root)
    print(
        {
            "summary": f"{args.output_root}/viser_scene_index.json",
            "visualization_status": payload["summary"]["visualization_status"],
            "gate": payload["summary"]["gate"],
            "scene_count": payload["summary"]["scene_count"],
            "bookmarked_screenshot_count": payload["summary"]["bookmarked_screenshot_count"],
        }
    )


if __name__ == "__main__":
    main()
