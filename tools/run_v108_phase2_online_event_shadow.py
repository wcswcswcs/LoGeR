#!/usr/bin/env python3
"""Generate v108 Phase2 shadow online events from a candidate summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v108.online_events import write_online_event_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    result = write_online_event_artifacts(
        case_name=args.case_name,
        source_summary_path=REPO_ROOT / args.source_summary,
        output_root=REPO_ROOT / args.output_root,
        accepted_reference_inputs=[],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    audit = result.get("reference_path_access_audit", {})
    if audit.get("status") != "PASS_NO_REFERENCE_INPUTS":
        return 2
    if result.get("reference_label_used_count") != 0:
        return 2
    if not result.get("all_events_shadow_only", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
