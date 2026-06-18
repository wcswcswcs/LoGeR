from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.role_gated_geometry_optimization import geometry_stage_decision
from stream4d_native.v37_object_field_adapter import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-summary", default="outputs/audit/v43_2_full_matching_significance/full_matching_significance_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_geometry_optimization_diagnostic")
    args = parser.parse_args()
    stage1 = read_json(ROOT / args.stage1_summary) or {}
    payload = geometry_stage_decision(stage1)
    out = ROOT / args.output_root
    write_json(out / "geometry_optimization_diagnostic_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
