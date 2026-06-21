from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.eval_aligned_ap_bridge import EvalAlignmentManifest, validate_eval_alignment_manifest
from stream4d_native.stage1_scale_aware_typed_assembly import read_json, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v45 eval-aligned AP diagnostic status.")
    parser.add_argument("--stage1-summary", default="outputs/audit/v45_stage1_full/stage1_full_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v45_eval_aligned_ap")
    args = parser.parse_args()
    stage1 = read_json(ROOT / args.stage1_summary) or {}
    manifest = EvalAlignmentManifest(
        alignment_protocol="scene_level_eval_sim3_blocked_until_stage1_pass",
        uses_gt_for_prediction=False,
        uses_gt_for_evaluation_alignment=True,
        scale_aligned_eval_protocol=True,
        is_method_result=False,
    )
    payload = {
        "phase": "v45_eval_aligned_ap",
        "created_at": utc_now(),
        "status": "blocked_stage1_not_method" if not stage1.get("stage1_run_as_method") else "not_run",
        "AP": None,
        "AP50": None,
        "AP25": None,
        "alignment_manifest": manifest.__dict__,
        "manifest_gate": validate_eval_alignment_manifest(manifest),
        "gate": {"pass": False, "ap_bridge_status": "BLOCKED_STAGE1_NOT_METHOD" if not stage1.get("stage1_run_as_method") else "NOT_RUN"},
    }
    out = ROOT / args.output_root
    write_json(out / "eval_aligned_ap_summary.json", payload)
    print(json.dumps({"summary": str(out / "eval_aligned_ap_summary.json"), "status": payload["status"], "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

