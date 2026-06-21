from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v46_signed_mask_graph import eval_aligned_ap_policy, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v46 eval-aligned AP policy diagnostic.")
    parser.add_argument("--stage1-root", default="outputs/audit/v46_full_stage1")
    parser.add_argument("--output-root", default="outputs/audit/v46_eval_aligned_ap")
    args = parser.parse_args()
    stage1 = read_json(ROOT / args.stage1_root / "full_stage1_controls.json") or {}
    payload = eval_aligned_ap_policy(stage1)
    out = ROOT / args.output_root
    write_json(out / "eval_aligned_ap_summary.json", payload)
    print(json.dumps({"summary": str(out / "eval_aligned_ap_summary.json"), "status": payload["status"], "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
