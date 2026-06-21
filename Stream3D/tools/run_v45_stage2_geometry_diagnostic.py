from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.stage1_scale_aware_typed_assembly import read_json, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v45 Stage-2 eligibility diagnostic.")
    parser.add_argument("--stage1-summary", default="outputs/audit/v45_stage1_full/stage1_full_summary.json")
    parser.add_argument("--scale-summary", default="outputs/audit/v45_scale_alignment/scale_alignment_audit.json")
    parser.add_argument("--output-root", default="outputs/audit/v45_stage2_geometry")
    args = parser.parse_args()
    stage1 = read_json(ROOT / args.stage1_summary) or {}
    scale = read_json(ROOT / args.scale_summary) or {}
    allowed = bool(stage1.get("gate", {}).get("pass") and scale.get("gate", {}).get("pass"))
    payload = {
        "phase": "v45_stage2_geometry_eligibility",
        "created_at": utc_now(),
        "entry_condition": {
            "phase6_stage1_gate_passed": bool(stage1.get("gate", {}).get("pass")),
            "phase1_scale_alignment_pass_or_blocked": bool(scale.get("gate", {}).get("pass")),
        },
        "stage2_allowed": allowed,
        "status": "STAGE2_BLOCKED" if not allowed else "STAGE2_DIAGNOSTIC_ALLOWED",
        "reason": "Stage-1 and scale guard must both pass before Stage-2 can be mainline.",
        "gate": {"pass": allowed},
    }
    out = ROOT / args.output_root
    write_json(out / "stage2_geometry_eligibility.json", payload)
    print(json.dumps({"summary": str(out / "stage2_geometry_eligibility.json"), "status": payload["status"], "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

