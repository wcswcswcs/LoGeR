from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import read_json, stage2_geometry_diagnostic, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check v44 Stage-2 geometry eligibility.")
    parser.add_argument("--strategy-summary", default="outputs/audit/v44_strategy_comparison/strategy_comparison.json")
    parser.add_argument("--controls-summary", default="outputs/audit/v44_controls/controls_and_significance.json")
    parser.add_argument("--output-root", default="outputs/audit/v44_stage2_geometry")
    args = parser.parse_args()
    strategies = read_json(ROOT / args.strategy_summary) or {}
    controls = read_json(ROOT / args.controls_summary) or {}
    payload = stage2_geometry_diagnostic(strategies, controls)
    out = ROOT / args.output_root
    write_json(out / "stage2_geometry_eligibility.json", payload)
    print(json.dumps({"summary": str(out / "stage2_geometry_eligibility.json"), "status": payload["status"], "stage2_allowed": payload["stage2_allowed"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
