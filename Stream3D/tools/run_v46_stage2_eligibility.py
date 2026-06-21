from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import read_json, stage2_eligibility, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check v46 Stage-2 eligibility.")
    parser.add_argument("--stage1-root", default="outputs/audit/v46_full_stage1")
    parser.add_argument("--fact-root", default="outputs/audit/v46_fact_lock")
    parser.add_argument("--output-root", default="outputs/audit/v46_stage2")
    args = parser.parse_args()
    stage1 = read_json(ROOT / args.stage1_root / "full_stage1_controls.json") or {}
    fact = read_json(ROOT / args.fact_root / "fact_lock.json") or {}
    payload = stage2_eligibility(stage1, fact)
    out = ROOT / args.output_root
    write_json(out / "stage2_eligibility.json", payload)
    print(json.dumps({"summary": str(out / "stage2_eligibility.json"), "status": payload["status"], "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
